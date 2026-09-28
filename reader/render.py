"""Rendering for the Skimmer reader (HTML + terminal).

One token system powers every page (form / result / plan / status). Light and
dark themes are driven by `data-theme` on <html>, set before first paint from
localStorage or the system preference; the header toggle flips it and the
choice sticks. The four importance bands are the only chromatic content on the
result page.
"""

from __future__ import annotations

from .i18n import L, js_i18n
from .scoring import SentenceScore

BRAND = "Skimmer"
SLOGAN = "让重点自己浮出来"
SLOGAN_EN = "Important sentences, surfaced."

THEME_BOOT = """<script>
(function(){try{var t=localStorage.getItem('skimmer-theme');
if(t!=='light'&&t!=='dark'){t=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}
document.documentElement.dataset.theme=t;}catch(e){document.documentElement.dataset.theme='light';}
try{if(localStorage.getItem('skimmer-focus')==='1'){document.documentElement.classList.add('focus-mode');}}catch(e){}
try{var g=localStorage.getItem('skimmer-lang');if(g!=='zh'&&g!=='en'){g='en';}
document.documentElement.dataset.lang=g;document.documentElement.lang=g;}catch(e){document.documentElement.dataset.lang='en';}})();
</script>"""

UI_JS = """<script>
var I18N = __I18N__;
(function(){
var root=document.documentElement;
function lang(){return root.dataset.lang==='zh'?'zh':'en';}
function t(k){var e=I18N[k];return e?e[lang()]:k;}
function cur(){return root.dataset.theme==='dark'?'dark':'light';}
function focusOn(){return root.classList.contains('focus-mode');}
function apply(){
  root.lang=lang();
  var lt=document.getElementById('lang-toggle');
  if(lt){lt.textContent=t(lang()==='zh'?'btn.lang.to_en':'btn.lang.to_zh');lt.title=t('title.lang');}
  var th=document.getElementById('theme-toggle');
  if(th){th.textContent=cur()==='dark'?t('btn.theme.light'):t('btn.theme.dark');th.title=t('title.theme');
         th.setAttribute('aria-pressed',cur()==='dark'?'true':'false');}
  var fo=document.getElementById('focus-toggle');
  if(fo){fo.textContent=focusOn()?t('btn.focus.off'):t('btn.focus.on');fo.title=t('title.focus');
         fo.setAttribute('aria-pressed',focusOn()?'true':'false');}
  document.querySelectorAll('[data-i18n-ph]').forEach(function(el){el.placeholder=t(el.getAttribute('data-i18n-ph'));});
  document.querySelectorAll('[data-i18n-title]').forEach(function(el){el.title=t(el.getAttribute('data-i18n-title'));});
}
var lt=document.getElementById('lang-toggle');
if(lt){lt.addEventListener('click',function(){var n=lang()==='zh'?'en':'zh';root.dataset.lang=n;
try{localStorage.setItem('skimmer-lang',n);}catch(e){}apply();});}
var th=document.getElementById('theme-toggle');
if(th){th.addEventListener('click',function(){var n=cur()==='dark'?'light':'dark';root.dataset.theme=n;
try{localStorage.setItem('skimmer-theme',n);}catch(e){}apply();});}
var fo=document.getElementById('focus-toggle');
if(fo){fo.addEventListener('click',function(){var on=!focusOn();root.classList.toggle('focus-mode',on);
try{localStorage.setItem('skimmer-focus',on?'1':'0');}catch(e){}apply();});}
document.addEventListener('keydown',function(e){var tag=(e.target&&e.target.tagName)||'';
if(!e.metaKey&&!e.ctrlKey&&!e.altKey&&!/^(INPUT|TEXTAREA|SELECT)$/.test(tag)){
 if(e.key==='t'||e.key==='T'){if(th)th.click();}
 if(e.key==='l'||e.key==='L'){if(lt)lt.click();}
 if(e.key==='f'||e.key==='F'){if(fo)fo.click();}
}});
apply();})();
</script>"""
UI_JS = UI_JS.replace("__I18N__", js_i18n())


PAGE_CSS = """
:root {
  --bg: #f6f7f8;
  --surface: #ffffff;
  --surface-2: #eef0f2;
  --ink: #17191c;
  --muted: #59616e;
  --faint: #98a1ad;
  --border: #e2e5e9;
  --accent: #0f766e;
  --accent-ink: #0b5d57;
  --on-accent: #ffffff;
  --b3-bar: #c8443b; --b3-tint: #fde7e5;
  --b2-bar: #c07d2a; --b2-tint: #fcefdb;
  --b1-bar: #b09a2f; --b1-tint: #fbf7de;
  --ok-bg: #e6f4ea; --ok-fg: #137333;
  --warn-bg: #fdf3e0; --warn-fg: #8a5a0c;
  --fail-bg: #fde7e5; --fail-fg: #b3372c;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
          "Microsoft YaHei", "Noto Sans SC", sans-serif;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  color-scheme: light;
}
:root[data-theme="dark"] {
  --bg: #131518;
  --surface: #1a1d21;
  --surface-2: #20242a;
  --ink: #e8eaed;
  --muted: #a3abb5;
  --faint: #6e7782;
  --border: #2b3037;
  --accent: #2dd4bf;
  --accent-ink: #5eead4;
  --on-accent: #05231f;
  --b3-bar: #f07070; --b3-tint: rgba(240, 112, 112, .14);
  --b2-bar: #e0a13c; --b2-tint: rgba(224, 161, 60, .13);
  --b1-bar: #c9bd4e; --b1-tint: rgba(201, 189, 78, .12);
  --ok-bg: rgba(52, 199, 123, .15); --ok-fg: #7ee2a8;
  --warn-bg: rgba(240, 178, 74, .14); --warn-fg: #e8c07a;
  --fail-bg: rgba(240, 112, 112, .14); --fail-fg: #f0a9a3;
  color-scheme: dark;
}
.lang-zh { display: none; }
html[data-lang="zh"] .lang-zh { display: inline; }
html[data-lang="zh"] .lang-en { display: none; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 400 17px/1.85 var(--sans); -webkit-font-smoothing: antialiased;
       text-rendering: optimizeLegibility; }
header.top { position: sticky; top: 0; z-index: 10;
             display: flex; justify-content: space-between; align-items: center;
             gap: 16px; padding: 10px 24px; background: var(--bg);
             border-bottom: 1px solid var(--border); }
@supports (backdrop-filter: blur(8px)) {
  header.top { background: color-mix(in srgb, var(--bg) 86%, transparent);
               backdrop-filter: blur(10px); }
}
.brand { display: flex; align-items: baseline; gap: 10px; text-decoration: none;
         color: var(--ink); white-space: nowrap; }
.brand .name { font-weight: 700; font-size: 17px; letter-spacing: -0.01em; }
.brand .slogan { color: var(--muted); font-size: 12.5px; }
@media (max-width: 640px) { .brand .slogan { display: none; } }
header nav { display: flex; align-items: center; gap: 2px; }
header nav a { color: var(--muted); text-decoration: none; font-size: 13.5px;
               padding: 6px 10px; border-radius: 8px; }
header nav a:hover { color: var(--ink); background: var(--surface-2); }
#theme-toggle { margin-left: 8px; padding: 5px 13px; border: 1px solid var(--border);
                border-radius: 999px; background: var(--surface); color: var(--muted);
                font: 500 13px var(--sans); cursor: pointer; }
#theme-toggle:hover { color: var(--ink); border-color: var(--accent); }
main { max-width: 860px; margin: 0 auto; padding: 30px 24px 88px; }
h1 { font-size: clamp(1.5rem, 2.4vw, 1.9rem); letter-spacing: -0.02em;
     line-height: 1.25; margin: 2px 0 8px; font-weight: 650; }
p, li { max-width: 68ch; }
.lede { color: var(--muted); margin: 0 0 20px; max-width: 62ch; }
.doc-sub { color: var(--muted); font-size: 14px; margin: 0 0 4px; }
.stats { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 2px; }
.stat { display: inline-flex; align-items: baseline; gap: 6px; padding: 5px 12px;
        border: 1px solid var(--border); border-radius: 999px;
        background: var(--surface); color: var(--muted); font-size: 12.5px; }
.stat b { color: var(--ink); font-weight: 600; font-size: 13px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 14px 0 24px;
          font-size: 12.5px; color: var(--muted); }
.legend span { padding: 2px 0 2px 9px; border-left: 3px solid transparent; }
.reading { max-width: 68ch; }
.sentence { margin: 0 0 8px; padding: 9px 15px 9px 14px;
            border-left: 3px solid transparent; border-radius: 0 8px 8px 0;
            line-height: 1.85; }
.sentence[data-band="3"] { background: var(--b3-tint); border-left-color: var(--b3-bar); }
.sentence[data-band="2"] { background: var(--b2-tint); border-left-color: var(--b2-bar); }
.sentence[data-band="1"] { background: var(--b1-tint); border-left-color: var(--b1-bar); }
.sentence[data-band="0"] { color: var(--faint); }
html.focus-mode .sentence[data-band="0"], html.focus-mode .sentence[data-band="1"] { display: none; }
.chip-btn { padding: 5px 13px; border: 1px solid var(--border); border-radius: 999px;
            background: var(--surface); color: var(--muted); font: 500 12.5px var(--sans);
            cursor: pointer; }
.chip-btn:hover { color: var(--ink); border-color: var(--accent); }
html.focus-mode .chip-btn { color: var(--accent-ink); border-color: var(--accent); }
.actions { margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--border);
           display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
           color: var(--muted); font-size: 13.5px; }
.actions a { color: var(--accent-ink); text-decoration: none; }
.actions a:hover { text-decoration: underline; }
.progress { position: relative; margin: 14px 0 22px; padding: 14px 16px 17px;
            border: 1px solid var(--border); border-radius: 12px;
            background: var(--surface); color: var(--muted); font-size: 13.5px;
            overflow: hidden; }
.progress .track { position: absolute; left: 0; right: 0; bottom: 0; height: 3px;
                   background: var(--surface-2); }
.progress .bar { height: 100%; width: 4%; background: var(--accent);
                 transition: width 240ms ease; }
.err { margin-top: 18px; padding: 12px 16px; border: 1px solid var(--fail-fg);
       border-radius: 10px; background: var(--fail-bg); color: var(--fail-fg);
       font-size: 14px; }
.meta { color: var(--muted); font-size: 13.5px; line-height: 1.9; }
.meta b { color: var(--ink); font-weight: 500; }
a { color: var(--accent-ink); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
@media (prefers-reduced-motion: no-preference) {
  a, button, input, .progress .bar { transition: color 160ms ease, background-color 160ms ease,
                                     border-color 160ms ease; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
@media print {
  header.top, .actions, .progress, .legend, #theme-toggle { display: none !important; }
  body { background: #fff; color: #000; font-size: 12pt; }
  .sentence { background: transparent !important; border-left-color: #999 !important;
              page-break-inside: avoid; }
  main { max-width: none; padding: 0; }
}
"""

BANDS = {
    3: {"zh": "核心", "en": "Core"},
    2: {"zh": "重要", "en": "Important"},
    1: {"zh": "次要", "en": "Minor"},
    0: {"zh": "可略", "en": "Skip"},
    -1: {"zh": "未分析", "en": "Not analyzed"},
}


def _escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def header_html() -> str:
    return f"""<header class="top">
  <a class="brand" href="/"><span class="name">{BRAND}</span><span class="slogan">{L(SLOGAN, SLOGAN_EN)}</span></a>
  <nav><a href="/">{L("新建", "New")}</a><a href="/plan">{L("训练方案", "Training plan")}</a><a href="/status">{L("环境自检", "Status")}</a>
    <button id="lang-toggle" type="button">中文</button>
    <button id="theme-toggle" type="button" aria-pressed="false">Dark</button></nav>
</header>"""


def page_shell(title_html: str, doc_title: str = "", sub_html: str = "") -> str:
    doc = f"{_escape(doc_title)} · {BRAND}" if doc_title else BRAND
    sub = f'<p class="doc-sub">{sub_html}</p>' if sub_html else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{doc}</title>
{THEME_BOOT}
<style>{PAGE_CSS}</style>
</head>
<body>
{header_html()}
<main>
<h1>{title_html}</h1>
{sub}
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
        f'<span style="border-left-color: var(--b{b}-bar);">{L(BANDS[b]["zh"], BANDS[b]["en"])}</span>'
        if b >= 0 else f'<span>{L(BANDS[b]["zh"], BANDS[b]["en"])}</span>'
        for b in bands
    )
    stats = [
        f'<span class="stat">{L("已分析", "Analyzed")} <b>{len(scorable)}</b> {L("句", "sentences")}</span>',
        f'<span class="stat">{L("核心", "Core")} <b>{counts[3]}</b></span>',
        f'<span class="stat">{L("重要", "Important")} <b>{counts[2]}</b></span>',
        f'<span class="stat">{L("次要", "Minor")} <b>{counts[1]}</b></span>',
        f'<span class="stat">{L("可略", "Skip")} <b>{counts[0]}</b></span>',
    ]
    if analyzed is not None and analyzed < len(sentences):
        stats.append(
            '<span class="stat">'
            + L(
                f"共 {len(sentences)} 句，为控制等待只分析前 {analyzed} 句（可用 READER_MAX_SENTENCES 调整）",
                f"{len(sentences)} sentences total; first {analyzed} analyzed to bound latency "
                "(tune with READER_MAX_SENTENCES)",
            )
            + "</span>"
        )
    stats.append(
        '<button class="chip-btn" id="focus-toggle" type="button" aria-pressed="false">Key points only</button>'
    )
    paras = []
    for s in sentences:
        title = f' title="p={s.percentile:.2f}"' if s.band >= 0 else ""
        paras.append(f'<p class="sentence" data-band="{s.band}"{title}>{_escape(s.text)}</p>')
    return f"""<div class="stats">{"".join(stats)}</div>
<div class="legend">{chips}</div>
<div class="reading">{"".join(paras)}</div>
"""


def to_html(
    sentences: list[SentenceScore],
    title: str = "Reading result",
    goal: str = "",
    source: str = "",
    analyzed: int | None = None,
) -> str:
    sub = L(f"关注点：{goal}", f"Focus: {goal}") if goal else ""
    return (
        page_shell(_escape(title), doc_title=title, sub_html=sub)
        + result_body(sentences, goal=goal, source=source, analyzed=analyzed)
        + PAGE_FOOTER
    )


PAGE_FOOTER = f"""<div class="actions">
  <span>{L("按 <b>T</b> 深浅色 · <b>L</b> 语言 · <b>F</b> 只看重点；全部推理在本机完成，文本不外传",
            "Press <b>T</b> theme, <b>L</b> language, <b>F</b> focus; all inference runs locally, nothing leaves your machine")}</span>
  <a href="/">{L("返回，换一篇或换关注点", "Back: new text or focus")}</a>
</div>
</main>
{UI_JS}
</body></html>"""

ANSI = {3: "\033[41;97m", 2: "\033[43;30m", 1: "\033[103;30m", 0: "\033[90m", -1: ""}
RESET = "\033[0m"


def to_terminal(sentences: list[SentenceScore]) -> str:
    return "\n\n".join(f"{ANSI[s.band]}{s.text}{RESET}" for s in sentences)
