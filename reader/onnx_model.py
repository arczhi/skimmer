"""ONNX Runtime backend for the reader (cross-platform: CPU / CUDA / DirectML).

Runs the exported cross-encoder graph (input_ids + attention_mask -> logits)
and applies attention-pooling internally, so any machine that can run
onnxruntime can serve the reader — no Python model code, no GPU requirement.

Providers are auto-selected in this order:
    CUDA (NVIDIA) -> DirectML (any Windows GPU) -> CoreML (Apple) -> CPU
"""

from __future__ import annotations

import os

import numpy as np

# CUDA (NVIDIA) and DirectML (any Windows GPU) first; plain CPU is faster and
# more reliable than CoreML for these dynamic shapes on Mac.
PREFERRED_PROVIDERS = [
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
    "CoreMLExecutionProvider",
]


class OnnxDecisionModel:
    def __init__(self, onnx_path: str, tokenizer_dir: str | None = None, cpu_only: bool = False):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        available = ort.get_available_providers()
        if cpu_only:
            providers = ["CPUExecutionProvider"]
        else:
            providers = [p for p in PREFERRED_PROVIDERS if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]
        self._ort = ort
        self._onnx_path = onnx_path
        self.providers = providers
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.tokenizer_dir = tokenizer_dir or os.path.dirname(onnx_path)
        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def score_batch(self, prefix_ids: list[int], cand_ids: list[list[int]]) -> list[float]:
        seqs = [prefix_ids + c for c in cand_ids]
        width = max(len(s) for s in seqs)
        b = len(seqs)
        ids = np.zeros((b, width), dtype=np.int64)
        mask = np.zeros((b, width), dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        try:
            out = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        except Exception:  # some EPs (e.g. CoreML) reject dynamic shapes -> CPU fallback
            so = self._ort.SessionOptions()
            so.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = self._ort.InferenceSession(
                self._onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
            )
            self.providers = ["CPUExecutionProvider"]
            out = self.sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        return [float(v) for v in out]

    def decide(
        self,
        state: str,
        question: str,
        options: list[str],
        tokenizer=None,  # drop-in compatibility
        temperature: float = 1.0,
        max_state_tokens: int = 1024,
        max_question_tokens: int = 96,
        max_option_tokens: int = 96,
    ) -> list[tuple[str, float]]:
        s_ids = self.tok(state, truncation=True, max_length=max_state_tokens)["input_ids"]
        q_ids = self.tok(question, truncation=True, max_length=max_question_tokens)["input_ids"]
        prefix = s_ids + q_ids
        # the graph's sequence dimension is dynamic, but keep rows bounded
        cand_ids = [
            self.tok(o, truncation=True, max_length=max_option_tokens)["input_ids"]
            for o in options
        ]
        logits = np.array(self.score_batch(prefix, cand_ids), dtype=np.float64)
        z = logits / max(temperature, 1e-6)
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        order = np.argsort(-p)
        return [(options[i], float(p[i])) for i in order]

    def independent_scores(self, prefix_ids: list[int], cand_ids: list[list[int]]) -> list[float]:
        """Absolute per-candidate probabilities (for cross-chunk ranking)."""
        logits = np.array(self.score_batch(prefix_ids, cand_ids), dtype=np.float64)
        return [float(1.0 / (1.0 + np.exp(-v))) for v in logits]
