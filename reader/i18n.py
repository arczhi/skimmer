"""Bilingual UI strings for the Skimmer reader.

Server-rendered pages carry both languages inline (`L()` emits two spans that
CSS shows/hides by `data-lang` on <html>), so the toggle is instant and works
on POST responses too. Strings that live in attributes (placeholders, titles)
or are produced by JS are driven by `JS_I18N`.
"""

from __future__ import annotations

import json


def L(zh: str, en: str) -> str:
    """Inline dual-language span pair; default language (en) wins without JS."""
    return f'<span class="lang-zh">{zh}</span><span class="lang-en">{en}</span>'


JS_I18N = {
    "ph.text": {
        "zh": "粘贴要阅读的长文本（导入文件时此处可留空）",
        "en": "Paste a long text to read (leave empty when importing a file)",
    },
    "ph.goal": {
        "zh": "可留空；例如：关键数字 / 行动项 / 主要论点 / 风险",
        "en": "Optional; e.g. key numbers / action items / main argument / risks",
    },
    "ph.goal.off": {
        "zh": "当前模型按通用重要性打分",
        "en": "this model scores general importance",
    },
    "title.theme": {"zh": "切换深浅色（快捷键 T）", "en": "Switch light/dark (shortcut: T)"},
    "title.lang": {"zh": "切换到中文", "en": "Switch to English"},
    "title.focus": {"zh": "只显示核心与重要句", "en": "Show only core and important sentences"},
    "btn.theme.dark": {"zh": "深色", "en": "Dark"},
    "btn.theme.light": {"zh": "浅色", "en": "Light"},
    "btn.focus.on": {"zh": "只看重点", "en": "Key points only"},
    "btn.focus.off": {"zh": "显示全部", "en": "Show all"},
    "btn.lang.to_zh": {"zh": "中文", "en": "中文"},
    "btn.lang.to_en": {"zh": "EN", "en": "EN"},
}


def js_i18n() -> str:
    return json.dumps(JS_I18N, ensure_ascii=False)
