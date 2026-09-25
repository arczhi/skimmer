# skimmer

A local-first reading tool. Paste text or import a document, and every sentence
is scored for importance and painted into four bands: core, important, minor,
and skippable (dimmed). Your attention goes where it is needed.

All inference runs on your machine. Your text never leaves it.

## How it works

skimmer runs a decision model, not a summarizer. A small transformer reads
`state + question + candidate` and returns a probability for each candidate:

```
P(sentence | document, "which sentence is most important for understanding this text")
```

Two consequences that matter:

- **The goal is a runtime input.** Change the question (`key numbers`,
  `action items`, `risks`) and the highlights change. No retraining.
- **The output is a calibrated probability**, not a generated sentence, so the
  four bands reflect confidence rather than decoration.

Long documents are split into chunks, scored per chunk, then mapped to
document-level percentiles so cross-chunk ranks stay comparable.

## Quick start

```bash
git clone https://github.com/arczhi/skimmer.git
cd skimmer
pip install -r requirements.txt
python download_model.py      # 153 MB int8 model from the GitHub release
python check_windows.py       # verify environment, model files, providers
python app.py                 # http://127.0.0.1:8899
```

Windows: `pip install -r requirements.txt`, then `run_windows.bat`.
macOS and Linux: `python app.py`.

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
`batch` when the native Apple models are present, otherwise `onnx`.

| Backend | Model | Platform | Measured |
|---|---|---|---|
| `onnx` | distilled student, 150M, int8 153 MB | Windows, Linux, macOS (CPU / CUDA / DirectML) | 131 ms per sentence |
| `mlx` | decision model, 1.7B | Apple Silicon | 117 ms per sentence |
| `batch` | same, batched candidates | Apple Silicon | 75 ms per sentence |
| `batch4bit` | same, 4-bit 934 MB | Apple Silicon | 83 ms per sentence |

ONNX Runtime selects its execution provider automatically: CUDA (NVIDIA),
DirectML (any Windows GPU), then CPU. Install `onnxruntime-gpu` or
`onnxruntime-directml` to enable the GPU paths.

The repository ships code. Model weights live in the
[release](https://github.com/arczhi/skimmer/releases) and are fetched by
`download_model.py`:

- `model.int8.onnx` (153 MB) plus tokenizer files, used by all platforms
- `model.onnx` (606 MB, optional) for `--fp32`

Native Apple models are not included. Build them from the training repository
(`decision-model`, see below) if you want the larger model locally.

## Environment check

```bash
python check_windows.py
```

Checks Python version, dependencies, ONNX model files, execution providers and
runs one end-to-end inference. The same report is available in the browser at
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
│   ├── scoring.py           sentence split, chunking, scoring, percentile bands
│   ├── documents.py         pdf / docx / doc / txt import
│   ├── render.py            design tokens and four-band HTML rendering
│   ├── onnx_model.py        ONNX Runtime backend
│   ├── mlx_batch.py         Apple MLX batched backend
│   └── diagnostics.py       shared environment checks
├── plan.html                in-app design and training plan page
└── requirements.txt
```

## Model provenance

The ONNX student is a 150M ModernBERT trained by distillation:

- teacher: a 45-task decision model (Qwen3.5-2B) that scored 5,347 articles
  sentence by sentence
- student: cross-encoder over `state + question + sentence`, trained with
  knowledge distillation, task replay and a belief calibration objective
- export: ONNX with dynamic sequence lengths, then dynamic int8 quantization,
  verified against the reference implementation to 2e-5

Training code, configurations and the full experiment log live in the companion
repository (decision-model). The student currently agrees with its teacher on
the top-ranked sentence for 18% of held-out articles; a v2 run with five times
the salience data is in progress.

## Limits

- Context: 256 tokens for the 1.7B model, 1024 for the student. Long documents
  are chunked, so importance is judged inside a window, not globally.
- Cost: every sentence needs one forward pass. A 100 sentence document takes
  about 13 seconds on CPU and 2 to 3 seconds on a discrete GPU.
- Sentence splitting is rule based. Tables and code blocks need post-processing.
- The student is a speed and portability trade. For maximum quality use the
  native Apple backend with the 1.7B model.

## License

MIT for the code. The student model is derived from ModernBERT (Apache-2.0).
