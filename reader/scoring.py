"""Sentence-importance scoring backends (all models ship with the release).

Two scoring strategies:

- `salience`: the document is packed into segments and scored in one batched
  encoder pass; every sentence gets an absolute importance logit.
- `onnx` / `batch` / `batch4bit`: the goal-conditioned cross-encoder. A chunk of
  sentences is scored by asking "Which sentence is most important for <goal>?"
  with the chunk's sentences as runtime candidates; scores are softmaxed within
  the chunk and mapped to document-level percentiles.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


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


def _find_salience_path(root: str) -> str | None:
    """Whole-document model dir (models_onnx/salience_*), not the old cross-encoder."""
    base = os.path.join(root, "models_onnx")
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base)):
        if not name.startswith("salience_"):
            continue
        p = os.path.join(base, name, "model.int8.onnx")
        if os.path.exists(p):
            return p
    return None


class ImportanceScorer:
    """Backends: 'salience' | 'onnx' | 'batch' | 'batch4bit' (see module docstring)."""

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
            elif _find_salience_path(root):
                backend = "salience"  # whole-document model (fast on CPU)
            elif os.path.exists(
                os.environ.get(
                    "READER_ONNX_PATH",
                    os.path.join(root, "models_onnx", "student_salience", "model.int8.onnx"),
                )
            ):
                backend = "onnx"
            else:
                backend = "salience"  # nothing installed yet; fail with a clear message
        self.temperature = temperature
        self.backend = backend
        if backend == "salience":
            from .salience_model import SalienceDocModel

            default_path = _find_salience_path(_reader_root()) or os.path.join(
                _reader_root(), "models_onnx", "salience_minilm12", "model.int8.onnx"
            )
            onnx_path = model_dir or os.environ.get("READER_SALIENCE_PATH", default_path)
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(
                    f"no salience model at {onnx_path}; run: python download_model.py"
                )
            self.model_dir = os.path.dirname(onnx_path)
            self.model = SalienceDocModel(
                onnx_path,
                self.model_dir,
                max_seg_tokens=int(os.environ.get("READER_SALIENCE_SEG_TOKENS", "512")),
            )
            self.tok = None
        elif backend == "onnx":
            from .onnx_model import OnnxDecisionModel

            default_path = os.path.join(
                _reader_root(), "models_onnx", "student_salience", "model.int8.onnx"
            )
            onnx_path = model_dir or os.environ.get("READER_ONNX_PATH", default_path)
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(
                    f"no cross-encoder model at {onnx_path}; run: python download_model.py --legacy"
                )
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
            raise ValueError(f"unknown backend {backend!r} (use salience | onnx | batch | batch4bit)")

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
        if self.backend == "salience":
            logits = self.model.score_sentences(analyzed)
            for i, v in enumerate(logits):
                raw[i] = float(v)
            if progress is not None:
                progress(len(analyzed), len(analyzed))
        else:
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
