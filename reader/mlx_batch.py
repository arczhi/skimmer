"""Batched decision-model inference for the reader (v5 tower, mlx-lm backend).

Scores all candidates of a chunk in ONE forward pass by batching the
[state; question; candidate] sequences (right-padded; causal attention means
real tokens never see the padding, and pooling is masked).

The original model dir / code in decision-model is untouched: this module only
reads models/v5_batched (prepared by prep_batched_model.py) and can also load a
4-bit quantized sibling produced by mlx_lm.convert.
"""

from __future__ import annotations

import os

import mlx.core as mx

NEG = -1e9


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0))))


class BatchDecisionModel:
    def __init__(self, model_dir: str, head_path: str | None = None):
        from mlx_lm import load

        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.model, self._mlx_tok = load(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if head_path is None:
            for cand in (
                os.path.join(model_dir, "decision_head.safetensors"),
                os.path.join(os.path.dirname(model_dir), "shared", "decision_head.safetensors"),
            ):
                if os.path.exists(cand):
                    head_path = cand
                    break
        if head_path is None or not os.path.exists(head_path):
            raise FileNotFoundError("decision_head.safetensors not found")
        self.head = dict(mx.load(head_path))

    def score_batch(self, prefix_ids: list[int], cand_ids: list[list[int]]) -> list[float]:
        """One forward for all candidates; returns raw logits (higher = better)."""
        seqs = [prefix_ids + c for c in cand_ids]
        width = max(len(s) for s in seqs)
        b = len(seqs)
        ids = mx.zeros((b, width), dtype=mx.int32)
        mask = mx.zeros((b, width))
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = mx.array(s, dtype=mx.int32)
            mask[i, : len(s)] = 1.0
        ids = mx.stop_gradient(ids)
        h = self.model.model(ids)  # [B, L, d]
        scores = (h @ self.head["pool.proj.weight"].T).squeeze(-1) + self.head["pool.proj.bias"]
        scores = mx.where(mask > 0, scores, mx.array(NEG, dtype=scores.dtype))
        w = mx.softmax(scores, axis=-1)
        pooled = (h * w[..., None]).sum(axis=1).astype(mx.float32)

        ln_w, ln_b = self.head["head.0.weight"], self.head["head.0.bias"]
        mu = pooled.mean(axis=-1, keepdims=True)
        var = ((pooled - mu) ** 2).mean(axis=-1, keepdims=True)
        x = (pooled - mu) / mx.sqrt(var + 1e-5) * ln_w + ln_b
        x = _gelu(x @ self.head["head.1.weight"].T + self.head["head.1.bias"])
        out = (x @ self.head["head.4.weight"].T + self.head["head.4.bias"]).squeeze(-1)
        mx.eval(out)
        return [float(v) for v in out]

    def decide(
        self,
        state: str,
        question: str,
        options: list[str],
        tokenizer=None,  # accepted for drop-in compatibility; unused
        temperature: float = 1.0,
        max_state_tokens: int = 256,
        max_question_tokens: int = 96,
        max_option_tokens: int = 64,
    ) -> list[tuple[str, float]]:
        s_ids = self.tok(state, truncation=True, max_length=max_state_tokens)["input_ids"]
        q_ids = self.tok(question, truncation=True, max_length=max_question_tokens)["input_ids"]
        c_ids = [
            self.tok(o, truncation=True, max_length=max_option_tokens)["input_ids"] for o in options
        ]
        logits = self.score_batch(s_ids + q_ids, c_ids)
        z = mx.array(logits) / temperature
        p = mx.softmax(z, axis=-1)
        mx.eval(p)
        return sorted(zip(options, p.tolist()), key=lambda t: -t[1])
