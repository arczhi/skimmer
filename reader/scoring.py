"""Sentence-importance scoring with the decision model (local, MLX).

A document is split into sentences, grouped into context chunks (the model's
state window), and each chunk is scored in ONE forward pass per sentence by
asking: "Which sentence is most important for <goal>?" with all chunk
sentences as runtime candidates. Scores are then mapped to document-level
percentiles so chunks are comparable.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

DECISION_MODEL_PATH = os.environ.get(
    "DECISION_MODEL_PATH", os.path.expanduser("~/coding/decision-model")
)
DEFAULT_MODEL_DIR = os.path.join(DECISION_MODEL_PATH, "models_mlx", "v5_mlx")

sys.path.insert(0, os.path.join(DECISION_MODEL_PATH, "mlx"))


@dataclass
class SentenceScore:
    text: str
    score: float  # model probability within its chunk
    percentile: float  # rank within the document, 0..1 (1 = most important)
    band: int  # 0 (least) .. 3 (most)


def split_sentences(text: str) -> list[str]:
    """Paragraph-aware sentence split for English and CJK text."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        # CJK terminators or Latin terminators (kept with the sentence)
        parts = re.split(r"(?<=[。！？；!?])\s*|(?<=[.!?])\s+(?=[A-Z\"'(])", para)
        for part in parts:
            part = part.strip()
            if len(part) >= 8:  # drop headings/fragments
                sentences.append(part)
    return sentences


def _approx_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"\b\w+\b", text))
    return int(cjk * 1.0 + latin * 1.4)


def _chunk(sentences: list[str], max_tokens: int = 160, max_sentences: int = 14) -> list[list[int]]:
    """Group sentence indices into context chunks for the model's state window."""
    chunks: list[list[int]] = []
    current: list[int] = []
    used = 0
    for i, s in enumerate(sentences):
        cost = _approx_tokens(s)
        if current and (used + cost > max_tokens or len(current) >= max_sentences):
            chunks.append(current)
            current, used = [], 0
        current.append(i)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def _reader_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImportanceScorer:
    """Backends: 'batch' (batched, reader-local tower) | 'batch4bit' | 'mlx' (original)."""

    def __init__(
        self,
        model_dir: str | None = None,
        temperature: float = 1.0,
        backend: str | None = None,
    ):
        backend = backend or os.environ.get("READER_BACKEND", "")
        if not backend:
            root = _reader_root()
            if os.path.isdir(os.path.join(root, "models", "v5_batched")):
                backend = "batch"  # native Apple backend with the larger model
            elif os.path.exists(
                os.environ.get(
                    "READER_ONNX_PATH",
                    os.path.join(root, "models_onnx", "student_salience", "model.int8.onnx"),
                )
            ):
                backend = "onnx"  # fresh clone / Windows path
            else:
                backend = "mlx"
        self.temperature = temperature
        self.backend = backend
        if backend == "onnx":
            from .onnx_model import OnnxDecisionModel

            default_path = os.path.join(
                _reader_root(), "models_onnx", "student_salience", "model.int8.onnx"
            )
            onnx_path = model_dir or os.environ.get("READER_ONNX_PATH", default_path)
            self.model_dir = os.path.dirname(onnx_path)
            self.model = OnnxDecisionModel(onnx_path, self.model_dir)
            self.tok = None
        elif backend in ("batch", "batch4bit"):
            from .mlx_batch import BatchDecisionModel

            default_dir = os.path.join(
                _reader_root(),
                "models",
                "v5_batched_4bit" if backend == "batch4bit" else "v5_batched",
            )
            self.model_dir = model_dir or os.environ.get("READER_BATCH_DIR", default_dir)
            self.model = BatchDecisionModel(self.model_dir)
            self.tok = None
        else:
            from decision_mlx import DecisionModelMLX  # decision-model repo
            from transformers import AutoTokenizer

            self.model_dir = model_dir or os.environ.get("READER_MODEL_DIR", DEFAULT_MODEL_DIR)
            self.tok = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = DecisionModelMLX(self.model_dir)

    def score(
        self,
        text: str,
        goal: str = "understanding this text",
        progress=None,
        max_sentences: int | None = None,
    ) -> tuple[list[SentenceScore], int]:
        """Score sentences. Returns (scores, n_sentences_total).

        progress: optional callback(done, total) called after every chunk.
        max_sentences: when set, only the first N sentences are analyzed; the
        rest are returned with band -1 (plain, not analyzed).
        """
        sentences = split_sentences(text)
        total = len(sentences)
        if not sentences:
            return [], 0
        analyzed = sentences[:max_sentences] if max_sentences else sentences
        question = f"Which sentence is most important for {goal}?"
        raw: dict[int, float] = {}
        chunk_list = _chunk(analyzed)
        done = 0
        for chunk in chunk_list:
            chunk_text = " ".join(analyzed[i] for i in chunk)
            opts = [analyzed[i] for i in chunk]
            results = self.model.decide(
                chunk_text, question, opts, self.tok, temperature=self.temperature
            )
            probs_by_text = dict(results)
            for i in chunk:
                raw[i] = probs_by_text[analyzed[i]]
            done += len(chunk)
            if progress is not None:
                progress(done, len(analyzed))

        order = sorted(raw, key=lambda i: raw[i])
        ranks = {idx: r / max(len(order) - 1, 1) for r, idx in enumerate(order)}
        out = []
        for i, s in enumerate(sentences):
            if i in raw:
                p = ranks[i]
                band = 3 if p >= 0.9 else 2 if p >= 0.7 else 1 if p >= 0.4 else 0
            else:
                p, band = -1.0, -1
            out.append(SentenceScore(text=s, score=raw.get(i, 0.0), percentile=p, band=band))
        return out, total

    def max_sentences(self) -> int:
        """Soft cap for very long documents (env: READER_MAX_SENTENCES)."""
        return int(os.environ.get("READER_MAX_SENTENCES", "1500"))
