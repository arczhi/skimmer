"""Color rendering for scored sentences (HTML + terminal).

Design tokens are shared through PAGE_CSS so every page (form / result / plan /
status) uses one palette and one type stack. One accent color only; the four
importance bands are the only chromatic content on the result page.
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
h2 { font-size: 1.15rem; letter-spacing: -0.01em; margin: 34px 0 8px; font-weight: 600; }
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
"""

BANDS = {
    3: {"bg": "#fde7e5", "border": "#c8443b", "label": "核心"},
    2: {"bg": "#fcefdb", "border": "#c07d2a", "label": "重要"},
    1: {"bg": "#fbf7de", "border": "#b09a2f", "label": "次要"},
    0: {"bg": "transparent", "border": "transparent", "label": "可略"},
}


def _escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _reading_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{PAGE_CSS}
  .reading {{ max-width: 68ch; }}
  .sentence {{ margin: 0 0 10px; padding: 7px 14px 7px 13px; border-left: 3px solid transparent;
               border-radius: 0 6px 6px 0; line-height: 1.95; }}
  .sentence[data-band="3"] {{ background: {BANDS[3]["bg"]}; border-left-color: {BANDS[3]["border"]}; }}
  .sentence[data-band="2"] {{ background: {BANDS[2]["bg"]}; border-left-color: {BANDS[2]["border"]}; }}
  .sentence[data-band="1"] {{ background: {BANDS[1]["bg"]}; border-left-color: {BANDS[1]["border"]}; }}
  .sentence[data-band="0"] {{ color: var(--faint); }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 14px 0 26px; font-size: 13px;
             color: var(--muted); }}
  .legend span {{ padding-left: 9px; border-left: 3px solid transparent; }}
  .actions {{ margin-top: 34px; font-size: 14px; }}
</style>
</head>
<body>
<header class="top">
  <span class="mark">重点阅读器</span>
  <nav><a href="/">新建</a><a href="/plan">训练方案</a><a href="/status">环境自检</a></nav>
</header>
<main>
{body}
<div class="actions"><a href="/">返回，换一篇或换关注点</a></div>
</main>
</body></html>"""


def to_html(
    sentences: list[SentenceScore],
    title: str = "重点标记",
    goal: str = "",
    source: str = "",
) -> str:
    chips = "".join(
        f'<span style="border-left-color:{BANDS[b]["border"]};'
        f'{"background:" + BANDS[b]["bg"] + ";" if b else ""}">{BANDS[b]["label"]}</span>'
        for b in (3, 2, 1, 0)
    )
    counts = {b: sum(1 for s in sentences if s.band == b) for b in (3, 2, 1, 0)}
    paras = "".join(
        f'<p class="sentence" data-band="{s.band}" title="p={s.percentile:.2f}">{_escape(s.text)}</p>'
        for s in sentences
    )
    body = f"""
<h1>{_escape(title)}</h1>
<p class="meta">来源：<b>{_escape(source or "粘贴文本")}</b><br>
关注点：<b>{_escape(goal or "理解全文")}</b><br>
{len(sentences)} 句 · 核心 {counts[3]} · 重要 {counts[2]} · 次要 {counts[1]} · 可略 {counts[0]}</p>
<div class="legend">{chips}</div>
<div class="reading">{paras}</div>"""
    return _reading_page(title, body)


ANSI = {3: "\033[41;97m", 2: "\033[43;30m", 1: "\033[103;30m", 0: "\033[90m"}
RESET = "\033[0m"


def to_terminal(sentences: list[SentenceScore]) -> str:
    return "\n\n".join(f"{ANSI[s.band]}{s.text}{RESET}" for s in sentences)
