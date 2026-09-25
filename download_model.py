"""Download skimmer model weights from the GitHub release (stdlib only).

    python download_model.py                # salience MiniLM-L12 int8 (34 MB, recommended)
    python download_model.py --modernbert   # salience ModernBERT int8 (152 MB, better quality)
    python download_model.py --legacy       # old cross-encoder student (153 MB)
    python download_model.py --legacy --fp32
    python download_model.py --mlx4bit      # Apple MLX v5 4-bit (968 MB, Apple Silicon)

Files land in models_onnx/<name>/ (ONNX) or models/<name>/ (MLX) where the
reader looks for them. Release assets are prefixed with the model directory
name (e.g. salience_minilm12.model.int8.onnx) because asset names are flat
inside one GitHub release.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

REPO = os.environ.get("SKIMMER_REPO", "arczhi/skimmer")
TAG = os.environ.get("SKIMMER_MODEL_TAG", "v1.1.0")
LEGACY_TAG = os.environ.get("SKIMMER_LEGACY_TAG", "v1.0.0")

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]
SHARED_HEAD = ("decision_head.safetensors", "decision_head.safetensors", "shared")

MODELS = {
    "minilm12": {
        "tag": TAG,
        "prefix": "salience_minilm12",
        "dir": "models_onnx/salience_minilm12",
        "files": ["model.int8.onnx"] + TOKENIZER_FILES,
    },
    "modernbert": {
        "tag": TAG,
        "prefix": "salience_modernbert",
        "dir": "models_onnx/salience_modernbert",
        "files": ["model.int8.onnx"] + TOKENIZER_FILES,
    },
    "legacy": {
        "tag": LEGACY_TAG,
        "dir": "models_onnx/student_salience",
        "files": ["model.int8.onnx"] + TOKENIZER_FILES + ["special_tokens_map.json"],
    },
    "mlx4bit": {
        "tag": TAG,
        "prefix": "v5_batched_4bit",
        "dir": "models/v5_batched_4bit",
        "files": ["model.safetensors", "model.safetensors.index.json", "config.json",
                  "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"],
        "extra": [SHARED_HEAD],
    },
}


def fetch(tag: str, name: str, dest: Path) -> None:
    url = f"https://github.com/{REPO}/releases/download/{tag}/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {dest.name} ...", end="", flush=True)

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            got = min(blocks * block_size, total)
            pct = 100 * got / total
            print(f"\r  {dest.name} {pct:5.1f}%  {got/1e6:6.1f}/{total/1e6:.1f} MB", end="")

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    print(f"\r  {dest.name} done ({dest.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="minilm12",
                    help="minilm12 (default) | modernbert | legacy | mlx4bit")
    ap.add_argument("--modernbert", action="store_true", help="shortcut for --model modernbert")
    ap.add_argument("--legacy", action="store_true", help="shortcut for --model legacy")
    ap.add_argument("--mlx4bit", action="store_true", help="shortcut for --model mlx4bit (Apple Silicon)")
    ap.add_argument("--fp32", action="store_true", help="also fetch the fp32 model (legacy only)")
    ap.add_argument("--dir", default=None, help="override target directory")
    args = ap.parse_args()

    which = ("mlx4bit" if args.mlx4bit
             else "modernbert" if args.modernbert else "legacy" if args.legacy else args.model)
    spec = MODELS[which]
    root = Path(__file__).resolve().parent
    out = Path(args.dir) if args.dir else root / spec["dir"]
    print(f"downloading {which} (tag {spec['tag']}) -> {out}")
    try:
        prefix = spec.get("prefix")
        for name in list(spec["files"]) + (["model.onnx", "model.fp32.onnx"] if args.fp32 else []):
            dest = out / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  {name} already present, skipping")
                continue
            remote = f"{prefix}.{name}" if prefix else name
            fetch(spec["tag"], remote, dest)
        for remote, name, sub in spec.get("extra", []):
            dest = root / "models" / sub / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  {name} already present, skipping")
                continue
            fetch(spec["tag"], remote, dest)
    except Exception as e:  # noqa: BLE001
        print(f"\n[error] {type(e).__name__}: {e}")
        print("mirror hint: downloads come from GitHub releases; if blocked, download the files manually")
        print(f"             from https://github.com/{REPO}/releases/tag/{spec['tag']} into {out}")
        return 1
    if which == "mlx4bit":
        print("model ready. run with: READER_BACKEND=batch4bit python app.py")
    else:
        print("model ready. next: python check_windows.py  (then run_windows.bat or python app.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
