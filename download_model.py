"""Download the decision model from the GitHub release (stdlib only).

    python download_model.py            # int8 model (153 MB, recommended)
    python download_model.py --fp32     # fp32 model (606 MB, optional)

Files land in models_onnx/student_salience/ where app.py expects them.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

REPO = os.environ.get("SKIMMER_REPO", "arczhi/skimmer")
TAG = os.environ.get("SKIMMER_MODEL_TAG", "v1.0.0")
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

FILES = {
    "int8": ["model.int8.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    "fp32": ["model.onnx"],
}


def fetch(url: str, dest: Path) -> None:
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
    ap.add_argument("--fp32", action="store_true", help="also fetch the fp32 model")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / "models_onnx" / "student_salience"))
    args = ap.parse_args()

    out = Path(args.dir)
    wanted = list(FILES["int8"]) + (FILES["fp32"] if args.fp32 else [])
    print(f"downloading {len(wanted)} file(s) from {BASE}")
    try:
        for name in wanted:
            dest = out / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  {name} already present, skipping")
                continue
            fetch(f"{BASE}/{name}", dest)
    except Exception as e:  # noqa: BLE001
        print(f"\n[error] {type(e).__name__}: {e}")
        print("mirror hint: downloads come from GitHub releases; if blocked, download the files manually")
        print(f"             from https://github.com/{REPO}/releases/tag/{TAG} into {out}")
        return 1
    print("model ready. next: python check_windows.py  (then run_windows.bat or python app.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
