"""Whole-document salience backend (ONNX): one batched pass scores every sentence.

The exported graph consumes packed segments:

    input_ids      [B, T]     int64
    attention_mask [B, T]     int64
    span_mask      [B, S, T]  float32   (1.0 covering a sentence's tokens)
    -> logits      [B, S]     float32

so the cost is one encoder pass per ~512-token segment instead of one
cross-encoder forward per sentence. Measured (M5 CPU, int8): 8000 Chinese
chars (7.9k tokens, 16 segments) in ~0.55 s with MiniLM-L6, ~14k tok/s.
"""

from __future__ import annotations

import os

import numpy as np

PREFERRED_PROVIDERS = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
    "CoreMLExecutionProvider",
]


def _find_default_model(root: str) -> str | None:
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


class SalienceDocModel:
    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: str | None = None,
        max_seg_tokens: int = 512,
        batch_segments: int = 8,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        available = ort.get_available_providers()
        providers = [p for p in PREFERRED_PROVIDERS if p in available] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.providers = [p for p in self.sess.get_providers()]
        self.onnx_path = onnx_path
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir or os.path.dirname(onnx_path))
        self.max_seg_tokens = max_seg_tokens
        self.batch_segments = batch_segments

    def _pack(self, sentences: list[str]):
        tok = self.tok
        ids_per_sent = []
        for s in sentences:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            ids_per_sent.append(ids if ids else [tok.unk_token_id or 0])

        cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
        body = self.max_seg_tokens - (1 if cls_id is not None else 0) - (1 if sep_id is not None else 0)
        segments: list[list[int]] = []
        spans: list[list[tuple[int, int, int]]] = []
        cur_ids: list[int] = []
        cur_spans: list[tuple[int, int, int]] = []

        def flush():
            nonlocal cur_ids, cur_spans
            if cur_spans:
                off = 1 if cls_id is not None else 0
                seg = ([cls_id] if cls_id is not None else []) + cur_ids + (
                    [sep_id] if sep_id is not None else []
                )
                segments.append(seg)
                spans.append([(gi, s + off, e + off) for gi, s, e in cur_spans])
            cur_ids, cur_spans = [], []

        for gi, st in enumerate(ids_per_sent):
            if len(st) > body:
                st = st[:body]
            if cur_ids and len(cur_ids) + len(st) > body:
                flush()
            s0 = len(cur_ids)
            cur_ids.extend(st)
            cur_spans.append((gi, s0, s0 + len(st)))
        flush()
        return segments, spans

    def score_sentences(self, sentences: list[str]) -> list[float]:
        """Return one logit per sentence (higher = more important)."""
        if not sentences:
            return []
        segments, spans = self._pack(sentences)
        out = [0.0] * len(sentences)
        for i in range(0, len(segments), self.batch_segments):
            seg_batch = segments[i : i + self.batch_segments]
            span_batch = spans[i : i + self.batch_segments]
            T = max(len(s) for s in seg_batch)
            S = max(len(s) for s in span_batch)
            ids = np.zeros((len(seg_batch), T), dtype=np.int64)
            mask = np.zeros((len(seg_batch), T), dtype=np.int64)
            smask = np.zeros((len(seg_batch), S, T), dtype=np.float32)
            for b, (seg, sp) in enumerate(zip(seg_batch, span_batch)):
                ids[b, : len(seg)] = seg
                mask[b, : len(seg)] = 1
                for si, (gi, s, e) in enumerate(sp):
                    smask[b, si, s:e] = 1.0
            logits = self.sess.run(
                ["logits"], {"input_ids": ids, "attention_mask": mask, "span_mask": smask}
            )[0]
            for b, sp in enumerate(span_batch):
                for si, (gi, _, _) in enumerate(sp):
                    out[gi] = float(logits[b, si])
        return out
