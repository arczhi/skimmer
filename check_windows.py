"""Environment self-check for the Local Focus Reader (run on any platform).

    python check_windows.py

Thin CLI wrapper around reader.diagnostics (the same checks shown at /status
in the web UI).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reader.diagnostics import collect_checks  # noqa: E402

LEVEL = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}


def main() -> int:
    checks = collect_checks(scorer=None, smoke=False)
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        print(f"{LEVEL[c.status]} {c.name:<{width}} {c.detail}")
        if c.fix:
            print(f"       {' ' * width} -> {c.fix}")
    fails = [c for c in checks if c.status == "fail"]
    warns = [c for c in checks if c.status == "warn"]
    print("=" * 62)
    if fails:
        print("ACTION NEEDED:")
        for c in fails:
            print(f"  - {c.name}: {c.fix or c.detail}")
        return 1
    if warns:
        print("WARNINGS (usually fine):")
        for c in warns:
            print(f"  - {c.name}: {c.detail}")
    print("ALL CHECKS PASSED - start with: run_windows.bat  (or READER_BACKEND=onnx python app.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
