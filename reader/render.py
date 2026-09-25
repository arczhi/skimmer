"""Color rendering for scored sentences (HTML + terminal).

Design tokens live in PAGE_CSS so every page (form / result / plan / status)
shares one palette and one type stack. The four importance bands are the only
chromatic content on the result page.

Band -1 means "not analyzed" (the document was longer than the configured cap):
those sentences render as plain text.
"""

from __future__ import annotations

from .scoring import SentenceScore

PAGE_CSS = """
:root {
  --bg: #f6f7f8;
  --surface: #ffffff;
  --ink: #17191c;
  --muted: #59616e;
  --faint: #98a1ad;
  --border: #e3e6e9;
  --accent: #0f766e;
  --accent-ink: #0b5d57;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
          "Microsoft YaHei", "Noto Sans SC", sans-serif;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 400 16px/1.75 var(--sans); -webkit-font-smoothing: antialiased; }
header.top { display: flex; justify-content: space-between; align-items: center;
             max-width: 820px; margin: 0 auto; padding: 22px 24px 0; }
header.top .mark { font-weight: 600; letter-spacing: -0.01em; }
header.top nav a { color: var(--muted); text-decoration: none; font-size: 14px; margin-left: 18px; }
header.top nav a:hover { color: var(--accent-ink); }
main { max-width: 820px; margin: 0 auto; padding: 26px 24px 72px; }
h1 { font-size: clamp(1.45rem, 2.6vw, 1.85rem); letter-spacing: -0.02em;
     line-height: 1.28; margin: 4px 0 10px; font-weight: 650; }
p, li { max-width: 68ch; }
.lede { color: var(--muted); margin: 0 0 18px; }
.meta { color: var(--muted); font-size: 13.5px; line-height: 1.9; }
.meta b { color: var(--ink); font-weight: 500; }
a { color: var(--accent-ink); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
@media (prefers-reduced-motion: no-preference) {
  a, button, input { transition: color 160ms ease, background-color 160ms ease,
                     border-color 160ms ease; }
}
.reading { max-width: 68ch; }
.sentence { margin: 0 0 10px; padding: 7px 14px 7px 13px; border-left: 3px solid transparent;
            border-radius: 0 6px 6px 0; line-height: 1.95; }
.sentence[data-band="3"] { background: #fde7e5; border-left-color: #c8443b; }
.sentence[data-band="2"] { background: #fcefdb; border-left-color: #c07d2a; }
.sentence[data-band="1"] { background: #fbf7de; border-left-color: #b09a2f; }
.sentence[data-band="0"] { color: var(--faint); }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 14px 0 26px; font-size: 13px;
          color: var(--muted); }
.legend span { padding-left: 9px; border-left: 3px solid transparent; }
.actions { margin-top: 34px; font-size: 14px; }
.progress { margin: 10px 0 18px; padding: 10px 14px; border: 1px solid var(--border);
            border-radius: 8px; background: var(--surface); color: var(--muted);
            font-size: 13.5px; }
"""

BANDS = {
    3: {"bg": "#fde7e5", "border": "#c8443b", "label": "核心"},
    2: {"bg": "#fcefdb", "border": "#c07d2a", "label": "重要"},
    1: {"bg": "#fbf7de", "border": "#b09a2f", "label": "次要"},
    0: {"bg": "transparent", "border": "transparent", "label": "可略"},
    -1: {"bg": "transparent", "border": "transparent", "label": "未分析"},
}

PAGE_FOOTER = """<div class="actions"><a href="/">返回，换一篇或换关注点</a></div>
</main>
</body></html>"""


def _escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def page_shell(title: str) -> str:
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header class="top">
  <span class="mark">重点阅读器</span>
  <nav><a href="/">新建</a><a href="/plan">训练方案</a><a href="/status">环境自检</a></nav>
</header>
<main>
<h1>{_escape(title)}</h1>
"""


def result_body(
    sentences: list[SentenceScore],
    goal: str = "",
    source: str = "",
    analyzed: int | None = None,
) -> str:
    scorable = [s for s in sentences if s.band >= 0]
    counts = {b: sum(1 for s in scorable if s.band == b) for b in (3, 2, 1, 0)}
    has_plain = any(s.band < 0 for s in sentences)
    bands = (3, 2, 1, 0, -1) if has_plain else (3, 2, 1, 0)
    chips = "".join(
        f'<span style="border-left-color:{BANDS[b]["border"]};'
        f'{"background:" + BANDS[b]["bg"] + ";" if b > 0 else ""}">{BANDS[b]["label"]}</span>'
        for b in bands
    )
    analyzed_line = ""
    if analyzed is not None and analyzed < len(sentences):
        analyzed_line = (
            f"<br>文档共 {len(sentences)} 句，为控制等待时间已分析前 {analyzed} 句"
            "（其余不参与标记，可用 READER_MAX_SENTENCES 调整）"
        )
    paras = []
    for s in sentences:
        title = f' title="p={s.percentile:.2f}"' if s.band >= 0 else ""
        paras.append(f'<p class="sentence" data-band="{s.band}"{title}>{_escape(s.text)}</p>')
    return f"""<p class="meta">来源：<b>{_escape(source or "粘贴文本")}</b><br>
关注点：<b>{_escape(goal or "理解全文")}</b><br>
已分析 {len(scorable)} 句 · 核心 {counts[3]} · 重要 {counts[2]} · 次要 {counts[1]} · 可略 {counts[0]}{analyzed_line}</p>
<div class="legend">{chips}</div>
<div class="reading">{"".join(paras)}</div>
"""


def to_html(
    sentences: list[SentenceScore],
    title: str = "重点标记",
    goal: str = "",
    source: str = "",
    analyzed: int | None = None,
) -> str:
    return (
        page_shell(title)
        + result_body(sentences, goal=goal, source=source, analyzed=analyzed)
        + PAGE_FOOTER
    )


ANSI = {3: "\033[41;97m", 2: "\033[43;30m", 1: "\033[103;30m", 0: "\033[90m", -1: ""}
RESET = "\033[0m"


def to_terminal(sentences: list[SentenceScore]) -> str:
    return "\n\n".join(f"{ANSI[s.band]}{s.text}{RESET}" for s in sentences)
