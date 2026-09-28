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


def _t(lang: str, zh: str, en: str) -> str:
    """zh | en | both (inline dual spans for the web UI)."""
    if lang == "zh":
        return zh
    if lang == "en":
        return en
    return f'<span class="lang-zh">{zh}</span><span class="lang-en">{en}</span>'


def collect_checks(scorer=None, smoke: bool = True, lang: str = "zh") -> list[Check]:
    checks: list[Check] = []

    # platform
    try:
        import multiprocessing

        cpus = multiprocessing.cpu_count()
    except Exception:  # noqa: BLE001
        cpus = 0
    _add(checks, _t(lang, "平台", "Platform"), "ok",
         _t(lang, f"{platform.system()} {platform.release()} ({platform.machine()}), {cpus} 核",
            f"{platform.system()} {platform.release()} ({platform.machine()}), {cpus} cores"))
    if sys.version_info < (3, 10):
        _add(checks, _t(lang, "Python 版本", "Python version"), "fail", sys.version.split()[0],
             _t(lang, "安装 Python 3.10+", "install Python 3.10+"))
    else:
        _add(checks, _t(lang, "Python 版本", "Python version"), "ok", sys.version.split()[0])

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
            gpu_hint = _t(lang,
                          "可选：pip install onnxruntime-gpu（NVIDIA）/ onnxruntime-directml（任意 GPU）",
                          "optional: pip install onnxruntime-gpu (NVIDIA) / onnxruntime-directml (any GPU)")
        _add(checks, _t(lang, "ONNX 执行后端", "ONNX providers"), "ok", ", ".join(providers), gpu_hint)
    except Exception as e:  # noqa: BLE001
        _add(checks, "onnxruntime", "fail", str(e), "pip install onnxruntime")

    if platform.system() == "Darwin":
        try:
            import mlx.core  # noqa: F401

            _add(checks, _t(lang, "mlx-lm（Apple MLX 后端）", "mlx-lm (Apple MLX backend)"), "ok",
                 _t(lang, "可用", "available"))
        except Exception:  # noqa: BLE001
            _add(checks, _t(lang, "mlx-lm（Apple MLX 后端）", "mlx-lm (Apple MLX backend)"), "warn",
                 _t(lang, "未安装", "not installed"),
                 _t(lang, "ONNX 后端仍可用；如需原生：pip install mlx mlx-lm",
                    "the ONNX backend still works; for native: pip install mlx mlx-lm"))

        v5 = reader_root() / "models" / "v5_batched_4bit"
        if v5.exists():
            safet = v5 / "model.safetensors"
            _add(checks, _t(lang, "MLX 4-bit 模型", "MLX 4-bit model"), "ok",
                 f"{safet.name} ({safet.stat().st_size/1e6:.0f} MB)")
            head = reader_root() / "models" / "shared" / "decision_head.safetensors"
            _add(checks, "  decision_head.safetensors",
                 "ok" if head.exists() else "fail",
                 "" if head.exists() else _t(lang, "缺失", "missing"),
                 "" if head.exists() else "python download_model.py --mlx4bit")
        else:
            _add(checks, _t(lang, "MLX 4-bit 模型", "MLX 4-bit model"), "warn",
                 _t(lang, "未找到 models/v5_batched_4bit/", "models/v5_batched_4bit/ not found"),
                 _t(lang, "可选（Apple Silicon）：python download_model.py --mlx4bit",
                    "optional (Apple Silicon): python download_model.py --mlx4bit"))

    # model files: whole-document salience model (preferred) + legacy cross-encoder
    spath = salience_path()
    if spath is not None and spath.exists():
        _add(checks, _t(lang, "整篇 salience 模型", "Whole-document salience model"), "ok",
             f"{spath.parent.name}/{spath.name} ({spath.stat().st_size/1e6:.0f} MB)")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            f = spath.parent / name
            _add(checks, f"  {name}", "ok" if f.exists() else "fail",
                 "" if f.exists() else _t(lang, "缺失", "missing"),
                 "" if f.exists() else _t(lang, "重新拷贝模型目录", "copy the model directory again"))
    else:
        _add(checks, _t(lang, "整篇 salience 模型", "Whole-document salience model"), "warn",
             _t(lang, "未找到 models_onnx/salience_*/model.int8.onnx",
                "models_onnx/salience_*/model.int8.onnx not found"),
             _t(lang, "python download_model.py（推荐，34 MB，纯 CPU 秒级）；否则回退 cross-encoder",
                "python download_model.py (recommended, 34 MB, sub-second on CPU); otherwise falls back to the cross-encoder"))

    path = onnx_path()
    if path.exists():
        _add(checks, _t(lang, "ONNX 模型文件（legacy）", "ONNX model file (legacy)"), "ok",
             f"{path.name} ({path.stat().st_size/1e6:.0f} MB)")
    else:
        _add(checks, _t(lang, "ONNX 模型文件（legacy）", "ONNX model file (legacy)"), "warn",
             _t(lang, f"未找到 {path}", f"not found: {path}"),
             _t(lang, "仅在使用 reader/scoring.py 的 onnx 后端时需要",
                "only needed for the onnx backend in reader/scoring.py"))

    # active backend + smoke inference
    if scorer is not None:
        model_dir = getattr(scorer, "model_dir", "?")
        providers = getattr(getattr(scorer, "model", None), "providers", None)
        detail = f"backend={scorer.backend} model={model_dir}"
        _add(checks, _t(lang, "当前推理后端", "Active backend"), "ok", detail,
             "" if not providers else f"providers: {', '.join(providers)}")
        if smoke:
            try:
                t0 = time.time()
                text = ("The company reported record revenue this quarter. "
                        "However, operating margins fell sharply. "
                        "Management reaffirmed guidance for the year.")
                res = scorer.score(text, goal="understanding this report")
                dt = time.time() - t0
                probs = ", ".join(f"{s.text.split()[0]}={s.score:.2f}" for s in sorted(res, key=lambda r: -r.percentile)[:3])
                _add(checks, _t(lang, "端到端推理（烟测）", "End-to-end inference (smoke)"), "ok",
                     _t(lang, f"{len(res)} 句 / {dt:.2f}s；top: {probs}",
                        f"{len(res)} sentences / {dt:.2f}s; top: {probs}"))
            except Exception as e:  # noqa: BLE001
                _add(checks, _t(lang, "端到端推理（烟测）", "End-to-end inference (smoke)"), "fail",
                     f"{type(e).__name__}: {e}",
                     _t(lang, "见 DESIGN/README 排错", "see DESIGN/README for troubleshooting"))
    return checks


STATUS_STYLE = {"ok": ("var(--ok-bg)", "var(--ok-fg)"),
                "warn": ("var(--warn-bg)", "var(--warn-fg)"),
                "fail": ("var(--fail-bg)", "var(--fail-fg)")}
STATUS_LABEL = {"ok": ("通过", "Pass"), "warn": ("注意", "Warn"), "fail": ("失败", "Fail")}


def checks_to_html(checks: list[Check], title: str = "Environment check") -> str:
    from .i18n import L
    from .render import PAGE_CSS, THEME_BOOT, UI_JS, header_html

    rows = []
    for c in checks:
        bg, fg = STATUS_STYLE[c.status]
        zh, en = STATUS_LABEL[c.status]
        fix = (f'<div style="color:var(--warn-fg);font-size:13px;margin-top:2px;">'
               f'{L("建议：", "Fix: ")}{c.fix}</div>') if c.fix else ""
        rows.append(
            f'<tr><td style="width:210px;">{c.name}</td>'
            f'<td><span style="background:{bg};color:{fg};padding:1px 8px;border-radius:10px;'
            f'font-size:12px;white-space:nowrap;">{L(zh, en)}</span></td>'
            f'<td>{c.detail}{fix}</td></tr>'
        )
    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")

    def _summary(zh: bool) -> str:
        if not fails and not warns:
            return "全部通过" if zh else "All checks passed"
        parts = []
        if fails:
            parts.append(f"失败 {fails} 项" if zh else f"{fails} failed")
        if warns:
            parts.append(f"警告 {warns} 项" if zh else f"{warns} warnings")
        return ("，" if zh else ", ").join(parts)

    summary = L(_summary(True), _summary(False))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} · Skimmer</title>
{THEME_BOOT}
<style>{PAGE_CSS}
 table {{ border-collapse:collapse; width:100%; font-size:14px; }}
 td {{ border-bottom:1px solid var(--border); padding:8px; vertical-align:top; }}
 td.name {{ width:220px; color:var(--ink); }}
 .chip {{ display:inline-block; padding:1px 9px; border-radius:999px; font-size:12px; }}
</style></head><body>
{header_html()}
<main>
<h1>{L("环境自检", title)} <span style="font-size:14px;color:var(--muted);">{summary}</span></h1>
<p class="meta">{L("全部推理在本机完成，文本不外传", "All inference runs locally; your text never leaves this machine")}</p>
<table>{"".join(rows)}</table>
</main>
{UI_JS}
</body></html>"""


def checks_as_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
