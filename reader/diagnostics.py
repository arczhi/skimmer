"""Environment / model diagnostics shared by the CLI self-check and the web UI."""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str = ""
    fix: str = ""


def _add(checks: list[Check], name: str, status: str, detail: str = "", fix: str = "") -> None:
    checks.append(Check(name, status, detail, fix))


def reader_root() -> Path:
    return Path(__file__).resolve().parent.parent


def salience_path() -> Path | None:
    base = reader_root() / "models_onnx"
    if base.is_dir():
        for p in sorted(base.iterdir()):
            if p.name.startswith("salience_") and (p / "model.int8.onnx").exists():
                return p / "model.int8.onnx"
    env = os.environ.get("READER_SALIENCE_PATH")
    return Path(env) if env else None


def onnx_path() -> Path:
    return Path(
        os.environ.get(
            "READER_ONNX_PATH",
            reader_root() / "models_onnx" / "student_salience" / "model.int8.onnx",
        )
    )


def collect_checks(scorer=None, smoke: bool = True) -> list[Check]:
    checks: list[Check] = []

    # platform
    try:
        import multiprocessing

        cpus = multiprocessing.cpu_count()
    except Exception:  # noqa: BLE001
        cpus = 0
    _add(checks, "平台", "ok", f"{platform.system()} {platform.release()} ({platform.machine()}), {cpus} 核")
    if sys.version_info < (3, 10):
        _add(checks, "Python 版本", "fail", sys.version.split()[0], "安装 Python 3.10+")
    else:
        _add(checks, "Python 版本", "ok", sys.version.split()[0])

    # dependencies
    try:
        import transformers

        _add(checks, "transformers", "ok", transformers.__version__)
    except Exception as e:  # noqa: BLE001
        _add(checks, "transformers", "fail", str(e), "pip install transformers")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        _add(checks, "onnxruntime", "ok", f"{ort.__version__}")
        gpu_hint = ""
        if "CUDAExecutionProvider" not in providers and "DmlExecutionProvider" not in providers:
            gpu_hint = "可选：pip install onnxruntime-gpu（NVIDIA）/ onnxruntime-directml（任意 GPU）"
        _add(checks, "ONNX 执行后端", "ok", ", ".join(providers), gpu_hint)
    except Exception as e:  # noqa: BLE001
        _add(checks, "onnxruntime", "fail", str(e), "pip install onnxruntime")

    if platform.system() == "Darwin":
        try:
            import mlx.core  # noqa: F401

            _add(checks, "mlx-lm（Apple MLX 后端）", "ok", "可用")
        except Exception:  # noqa: BLE001
            _add(checks, "mlx-lm（Apple MLX 后端）", "warn", "未安装", "ONNX 后端仍可用；如需原生：pip install mlx mlx-lm")

        v5 = reader_root() / "models" / "v5_batched_4bit"
        if v5.exists():
            safet = v5 / "model.safetensors"
            _add(checks, "MLX 4-bit 模型", "ok", f"{safet.name} ({safet.stat().st_size/1e6:.0f} MB)")
            head = reader_root() / "models" / "shared" / "decision_head.safetensors"
            _add(checks, "  decision_head.safetensors",
                 "ok" if head.exists() else "fail",
                 "" if head.exists() else "缺失",
                 "" if head.exists() else "python download_model.py --mlx4bit")
        else:
            _add(checks, "MLX 4-bit 模型", "warn", "未找到 models/v5_batched_4bit/",
                 "可选（Apple Silicon）：python download_model.py --mlx4bit")

    # model files: whole-document salience model (preferred) + legacy cross-encoder
    spath = salience_path()
    if spath is not None and spath.exists():
        _add(checks, "整篇 salience 模型", "ok",
             f"{spath.parent.name}/{spath.name} ({spath.stat().st_size/1e6:.0f} MB)")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            f = spath.parent / name
            _add(checks, f"  {name}", "ok" if f.exists() else "fail",
                 "" if f.exists() else "缺失", "" if f.exists() else "重新拷贝模型目录")
    else:
        _add(checks, "整篇 salience 模型", "warn", "未找到 models_onnx/salience_*/model.int8.onnx",
             "python download_model.py（推荐，34 MB，纯 CPU 秒级）；否则回退 cross-encoder")

    path = onnx_path()
    if path.exists():
        _add(checks, "ONNX 模型文件（legacy）", "ok", f"{path.name} ({path.stat().st_size/1e6:.0f} MB)")
    else:
        _add(checks, "ONNX 模型文件（legacy）", "warn", f"未找到 {path}",
             "仅在使用 reader/scoring.py 的 onnx 后端时需要")

    # active backend + smoke inference
    if scorer is not None:
        model_dir = getattr(scorer, "model_dir", "?")
        providers = getattr(getattr(scorer, "model", None), "providers", None)
        detail = f"backend={scorer.backend} model={model_dir}"
        _add(checks, "当前推理后端", "ok", detail, "" if not providers else f"providers: {', '.join(providers)}")
        if smoke:
            try:
                t0 = time.time()
                text = ("The company reported record revenue this quarter. "
                        "However, operating margins fell sharply. "
                        "Management reaffirmed guidance for the year.")
                res = scorer.score(text, goal="understanding this report")
                dt = time.time() - t0
                probs = ", ".join(f"{s.text.split()[0]}={s.score:.2f}" for s in sorted(res, key=lambda r: -r.percentile)[:3])
                _add(checks, "端到端推理（烟测）", "ok", f"{len(res)} 句 / {dt:.2f}s；top: {probs}")
            except Exception as e:  # noqa: BLE001
                _add(checks, "端到端推理（烟测）", "fail", f"{type(e).__name__}: {e}", "见 DESIGN/README 排错")
    return checks


STATUS_STYLE = {"ok": ("#e6f4ea", "#137333", "通过"), "warn": ("#fdf3e0", "#9a6510", "注意"), "fail": ("#fde7e5", "#b3372c", "失败")}


def checks_to_html(checks: list[Check], title: str = "环境自检") -> str:
    from .render import PAGE_CSS

    rows = []
    for c in checks:
        bg, fg, label = STATUS_STYLE[c.status]
        fix = f'<div style="color:#9a6510;font-size:13px;margin-top:2px;">建议：{c.fix}</div>' if c.fix else ""
        rows.append(
            f'<tr><td style="width:210px;">{c.name}</td>'
            f'<td><span style="background:{bg};color:{fg};padding:1px 8px;border-radius:10px;'
            f'font-size:12px;white-space:nowrap;">{label}</span></td>'
            f'<td>{c.detail}{fix}</td></tr>'
        )
    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")
    summary = (
        "全部通过" if not fails and not warns else
        f"{'失败 ' + str(fails) + ' 项' if fails else ''}{'，' if fails and warns else ''}{'警告 ' + str(warns) + ' 项' if warns else ''}"
    )
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>{PAGE_CSS}
 table {{ border-collapse:collapse; width:100%; font-size:14px; }}
 td {{ border-bottom:1px solid var(--border); padding:8px; vertical-align:top; }}
 td.name {{ width:220px; color:var(--ink); }}
 .chip {{ display:inline-block; padding:1px 9px; border-radius:999px; font-size:12px; }}
</style></head><body>
<header class="top"><span class="mark">重点阅读器</span>
 <nav><a href="/">新建</a><a href="/plan">训练方案</a></nav></header>
<main>
<h1>{title} <span style="font-size:14px;color:var(--muted);">{summary}</span></h1>
<p class="meta">全部推理在本机完成，文本不外传</p>
<table>{"".join(rows)}</table>
</main>
</body></html>"""


def checks_as_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
