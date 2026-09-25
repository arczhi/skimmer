# skimmer

A local-first reading tool. Paste text or import a document, and every sentence
is scored for importance and painted into four bands: core, important, minor,
and skippable (dimmed). Your attention goes where it is needed.

All inference runs on your machine. Your text never leaves it.

## How it works

skimmer scores a whole document in one batched encoder pass, not one forward per
sentence. The document is packed into aligned ~512-token segments, the encoder
(MiniLM-L12 or ModernBERT) reads each segment once, and every sentence's score
comes from mean-pooling its token hidden states through a small MLP head:

```
segment tokens -> encoder -> per-sentence span pooling -> importance logits
```

Scores are mapped to document-level percentiles so the four bands are stable
across documents.

Two runtimes are available:

- `salience` (default): the whole-document model above. Fast on pure CPU:
  8000 Chinese characters (7.9k tokens) in ~1.1 s, 8000 English characters in
  ~0.25 s (Apple M5, int8, ONNX Runtime CPU). The model is goal-free: it scores
  general importance.
- `onnx` / `batch`: the earlier goal-conditioned cross-encoder, where the
  question is a runtime input (`key numbers`, `action items`, `risks`) and
  changing it changes the highlights. Slower (one forward per sentence).

## Quick start

```bash
git clone https://github.com/arczhi/skimmer.git
cd skimmer
pip install -r requirements.txt
python download_model.py      # 34 MB int8 model from the GitHub release
python check_windows.py       # verify environment, model files, providers
python app.py                 # http://127.0.0.1:8899
```

Windows: `pip install -r requirements.txt`, then `run_windows.bat`.
macOS and Linux: `python app.py`.

One model is enough — the default download above is what most users want.
Optional variants (higher quality, Apple MLX) are listed under
[Which model should I download?](#which-model-should-i-download).

Command line:

```bash
python cli.py article.txt --goal "key numbers" --html out.html
python cli.py article.txt --terminal
cat article.txt | python cli.py - --goal "action items"
```

## Import formats

| Format | Parser | Notes |
|---|---|---|
| `.pdf` | pypdf | text PDFs; scanned pages need OCR first |
| `.docx` | python-docx | paragraphs and tables |
| `.doc` | antiword | optional; install antiword, or save as `.docx` |
| `.txt`, `.md` | built in | utf-8, utf-8-sig, gb18030 fallbacks |

Parsing happens locally. A missing parser or an unreadable file returns a clear
message instead of failing silently.

## Backends

`READER_BACKEND` selects the runtime. The default is picked automatically:
`batch` when local bf16 Apple models are present, then `salience` when a
whole-document model is installed, then `onnx`, else `salience`.

| Backend | Model | Platform | Measured |
|---|---|---|---|
| `salience` | whole-document MiniLM-L12, int8 34 MB | all (ONNX Runtime) | 8000 zh chars ~1.1 s; 8000 en chars ~0.25 s |
| `salience` | whole-document ModernBERT, int8 152 MB | all (ONNX Runtime) | 8000 en chars ~0.5 s; 8000 zh chars ~4.9 s |
| `onnx` | goal-conditioned cross-encoder, int8 153 MB | Windows, Linux, macOS (CPU / CUDA / DirectML) | 131 ms per sentence |
| `batch` | same, batched candidates | Apple Silicon; needs local bf16 weights (not released) | 75 ms per sentence |
| `batch4bit` | same, 4-bit 968 MB | Apple Silicon; from the release (`--mlx4bit`), opt in via `READER_BACKEND=batch4bit` | 83 ms per sentence |

Point `READER_SALIENCE_PATH` at a specific `model.int8.onnx` to choose between
the MiniLM and ModernBERT variants; `READER_SALIENCE_SEG_TOKENS` changes the
segment size (default 512).

ONNX Runtime selects its execution provider automatically: CUDA (NVIDIA),
DirectML (any Windows GPU), then CPU. Install `onnxruntime-gpu` or
`onnxruntime-directml` to enable the GPU paths.

## Which model should I download?

Most users need exactly one: the default.

| Download | Size | Get it if... |
|---|---|---|
| `python download_model.py` (MiniLM-L12) — **recommended** | 34 MB | you want the fast, small model. Stays around a second on 8000-character documents, even on a CPU-only laptop. |
| `python download_model.py --modernbert` | 152 MB | you want the highest teacher agreement and have CPU headroom or a GPU. On slower CPUs very long documents can take several seconds. |
| `python download_model.py --mlx4bit` | 968 MB | you are on Apple Silicon and want the goal-conditioned v5 model with native MLX inference (`READER_BACKEND=batch4bit`). |
| `python download_model.py --legacy` | 153 MB | you need the earlier goal-conditioned cross-encoder for compatibility. |

The reader picks a backend automatically from whatever is installed;
`READER_BACKEND` overrides it. You never need more than one model.

The repository ships code. Model weights live in the
[release](https://github.com/arczhi/skimmer/releases) and are fetched by
`download_model.py`:

- `salience_minilm12/model.int8.onnx` (34 MB) plus tokenizer files — default
- `salience_modernbert/model.int8.onnx` (152 MB) — optional, `--modernbert`
- `student_salience/model.int8.onnx` (153 MB) — legacy cross-encoder, `--legacy`
  (`model.onnx`, 606 MB, optional `--fp32`)
- `v5_batched_4bit/model.safetensors` (968 MB) + `shared/decision_head.safetensors`
  (34 MB) — Apple MLX, `--mlx4bit`

Asset names are flat inside a release, so they are prefixed with the model
directory name (`salience_minilm12.model.int8.onnx`,
`salience_minilm12.tokenizer.json`, ...). `download_model.py` maps them back to
the directories above; manual downloads should keep that convention.

The shipped Apple model is the 4-bit one; the bf16 tower (`batch` backend) is
not released. `skimmer` has no runtime dependency on the training repository
(`decision-model`) — it is credited below for provenance only.

## Environment check

```bash
python check_windows.py
```

Checks Python version, dependencies, model files, execution providers and runs
one end-to-end inference. The same report is available in the browser at
`/status`.

## Repository layout

```
skimmer/
├── app.py                   local web app (stdlib HTTP server, single threaded)
├── cli.py                   command line entry
├── download_model.py        fetch model weights from the release
├── check_windows.py         environment self check
├── run_windows.bat          Windows launcher
├── reader/
│   ├── scoring.py           sentence split, backend dispatch, percentile bands
│   ├── salience_model.py    whole-document ONNX backend (segment packing)
│   ├── documents.py         pdf / docx / doc / txt import
│   ├── render.py            design tokens and four-band HTML rendering
│   ├── onnx_model.py        ONNX Runtime backend (cross-encoder)
│   ├── mlx_batch.py         Apple MLX batched backend
│   └── diagnostics.py       shared environment checks
├── plan.html                in-app design and training plan page
└── requirements.txt
```

## Model provenance

The whole-document students were trained in the companion repository
(`decision-model`, `scripts/train_salience_doc.py`):

- teacher: a 45-task decision model (Qwen3.5-2B) that scored 5,347 articles
  sentence by sentence (soft probability distributions over candidate sentences)
- auxiliary data: CNN/DailyMail 3.0.0 with extractive ROUGE-1 labels
  (479k articles available, 200k indexed for training)
- students: MiniLM-L12 (33M) and ModernBERT-base (150M), full fine-tune, mixed
  KD (KL, T=2) + BCE, 20k steps
- agreement with the teacher on 347 held-out articles (top-1 / Spearman):

  | model | top-1 | Spearman | ROUGE AUC |
  |---|---|---|---|
  | legacy cross-encoder | 0.49 | 0.59 | — |
  | MiniLM-L12 whole-doc | 0.53 | 0.60 | 0.83 |
  | ModernBERT whole-doc | 0.57 | 0.68 | 0.84 |

- export: ONNX (segments, span pooling and head inside the graph), dynamic int8
  quantization; int8 agrees with the float reference within 0.01 top-1/Spearman
  and the file is 34 MB (MiniLM) / 152 MB (ModernBERT)

The legacy cross-encoder student remains available through `--legacy`; it is
goal-conditioned but pays one forward pass per sentence.

## Limits

- Context: each segment is ~512 tokens; sentences see their neighbours in the
  same segment. The legacy student used 1024-token chunks and the 1.7B model
  256 tokens.
- Cost: one encoder pass per segment. Measured for the whole-document MiniLM
  (ONNX, int8, CPU): 8000 English chars ~0.25 s, 8000 Chinese chars ~1.1 s.
  A 3000-sentence report is 60-80 segments, a few seconds on CPU.
- The whole-document backend is goal-free; the goal field is disabled in the UI
  when it is active. Use `READER_BACKEND=onnx` with the legacy model to score
  against a specific goal.
- Progress is streamed by segment; documents are still capped at 1,500 analyzed
  sentences by default. Set `READER_MAX_SENTENCES` to change the cap
  (0 disables it).
- Sentence splitting is rule based. Tables and code blocks need post-processing.
- The students are speed and portability trades. For maximum quality use the
  native Apple backend with the 1.7B model.

## License

MIT for the code. The student models are derived from MiniLM and ModernBERT
(Apache-2.0) and distilled from the decision-model teacher.
