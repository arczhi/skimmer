"""Local web reader: paste text or upload a document, get importance-colored output.

All inference runs locally (MLX on Apple Silicon, ONNX Runtime elsewhere);
uploaded files never leave the machine.

Usage:
    python app.py                # http://127.0.0.1:8765
    python app.py --port 9000
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from reader.diagnostics import checks_to_html, collect_checks
from reader.documents import DocumentError, extract_text
from reader.i18n import L
from reader.render import (BRAND, PAGE_CSS, PAGE_FOOTER, SLOGAN, SLOGAN_EN, THEME_BOOT,
                           UI_JS, header_html, page_shell, result_body)
from reader.scoring import ImportanceScorer

SCORER: ImportanceScorer | None = None


def default_doc() -> str:
    """The plan text pre-filled in the reader input for easy testing."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


def read_plan() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.html")
    if not os.path.exists(path):
        return "<html><body><p>plan.html not found.</p></body></html>"
    with open(path, encoding="utf-8") as f:
        return f.read()


FORM = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{brand} · {slogan_en}</title>
{theme_boot}
<style>{css}
  .compose {{ display: grid; gap: 18px; margin-top: 24px; }}
  textarea {{ width: 100%; min-height: 360px; resize: vertical; padding: 16px 18px;
             border: 1px solid var(--border); border-radius: 12px; background: var(--surface);
             color: var(--ink); font: 400 15.5px/1.8 var(--sans); }}
  textarea::placeholder {{ color: var(--muted); }}
  textarea:focus-visible, input[type=text]:focus-visible, input[type=file]:focus-visible {{
             outline: 2px solid var(--accent); outline-offset: 1px; border-color: transparent; }}
  .row {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
  .row label {{ font-size: 13px; color: var(--muted); min-width: 64px; }}
  input[type=text] {{ flex: 1; min-width: 220px; padding: 10px 12px; border: 1px solid var(--border);
             border-radius: 10px; background: var(--surface); color: var(--ink); font: 400 14px var(--sans); }}
  input[type=text]::placeholder {{ color: var(--muted); }}
  input[type=file] {{ font: 400 13px var(--sans); color: var(--muted); }}
  input[type=file]::file-selector-button {{ margin-right: 10px; padding: 8px 14px; border: 1px solid var(--border);
             border-radius: 10px; background: var(--surface); color: var(--ink); font: 500 13px var(--sans); cursor: pointer; }}
  input[type=file]::file-selector-button:hover {{ border-color: var(--accent); color: var(--accent-ink); }}
  button[type=submit] {{ justify-self: start; padding: 12px 28px; border: 0; border-radius: 10px;
            background: var(--accent); color: var(--on-accent); font: 600 15px var(--sans); cursor: pointer; }}
  button[type=submit]:hover {{ background: var(--accent-ink); color: var(--on-accent); }}
  .note {{ font-size: 13px; color: var(--muted); line-height: 1.7; }}
</style>
</head>
<body>
{header}
<main>
  <h1>{h1}</h1>
  <p class="lede">{lede}</p>
  {error}
  <form class="compose" method="post" action="/score" enctype="multipart/form-data">
    <textarea name="text" spellcheck="false" data-i18n-ph="ph.text"
      placeholder="Paste a long text to read (leave empty when importing a file)">{text}</textarea>
    <div class="row">
      <label for="file">{file_label}</label>
      <input id="file" type="file" name="file" accept=".pdf,.docx,.doc,.txt,.md,.markdown">
      <span class="note">{file_note}</span>
    </div>
    {goal_row}
    <button type="submit">{submit_label}</button>
  </form>
</main>
{theme_js}
</body></html>"""


def form_html(text: str = "", goal: str = "", error: str = "") -> str:
    return FORM.format(
        css=PAGE_CSS, brand=BRAND, slogan_en=SLOGAN_EN, theme_boot=THEME_BOOT, theme_js=UI_JS,
        header=header_html(), text=html.escape(text), lede=lede_text(), error=error,
        h1=L("粘贴文本，或导入文档", "Paste text or import a document"),
        file_label=L("文档", "Document"),
        file_note=L("支持 PDF / DOCX / DOC / TXT / Markdown，解析在本机完成",
                    "PDF / DOCX / DOC / TXT / Markdown; parsing happens locally"),
        submit_label=L("开始标记", "Skim it"),
        goal_row=goal_row(goal),
    )


def render_error(msg_html: str) -> str:
    return f'<div class="err">{msg_html}</div>'


def lede_text() -> str:
    if SCORER is not None and SCORER.backend == "salience":
        return L("整篇一次编码，纯 CPU 也秒级完成；重要的句子会自动浮出来。",
             "One pass over the whole document, sub-second even on CPU. The sentences that matter surface on their own.")
    return L("模型在本地给每句话打重要性分，四档着色；换一个关注点，重点就会变。",
             "Scores every sentence locally and paints four bands. Change the focus and the highlights change with it.")


def goal_row(goal: str) -> str:
    if SCORER is not None and SCORER.backend == "salience":
        return (
            f'<div class="row"><label for="goal">{L("关注点", "Focus")}</label>'
            '<input id="goal" type="text" value="" disabled data-i18n-ph="ph.goal.off" '
            'placeholder="this model scores general importance">'
            f'<span class="note">{L("整篇模型暂为通用重要性，关注点条件化在后续版本", "The whole-document model scores general importance; focus conditioning is on the roadmap")}</span></div>'
        )
    return (
        f'<div class="row"><label for="goal">{L("关注点", "Focus")}</label>'
        f'<input id="goal" type="text" name="goal" value="{html.escape(goal)}" '
        'data-i18n-ph="ph.goal" '
        'placeholder="Optional; e.g. key numbers / action items / main argument / risks"></div>'
    )


def render_result(text: str, goal: str, source: str) -> str:
    assert SCORER is not None
    sentences, total = SCORER.score(
        text, goal=goal or "understanding this text",
        max_sentences=SCORER.max_sentences(),
    )
    title_html = html.escape(source) if source else L("阅读结果", "Reading result")
    sub = L(f"关注点：{html.escape(goal or '理解全文')}",
            f"Focus: {html.escape(goal or 'understanding the text')}")
    return page_shell(title_html, doc_title=source, sub_html=sub) + result_body(
        sentences, goal=goal, source=source, analyzed=min(total, SCORER.max_sentences())
    ) + PAGE_FOOTER


def _parse_multipart(body: bytes, boundary: bytes) -> dict[str, bytes]:
    """Minimal multipart/form-data parser (stdlib cgi is gone in 3.13)."""
    out: dict[str, bytes] = {}
    for segment in body.split(b"--" + boundary):
        if not segment or segment in (b"--", b"--\r\n", b"\r\n"):
            continue
        head, sep, data = segment.partition(b"\r\n\r\n")
        if not sep:
            continue
        m = re.search(rb'name="([^"]+)"', head)
        if not m:
            continue
        name = m.group(1).decode("utf-8", "replace")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        out[name] = data
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _send(self, body: str, code: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/plan"):
            self._send(read_plan())
            return
        if self.path.startswith("/status"):
            checks = collect_checks(scorer=SCORER, smoke=SCORER is not None, lang="both")
            self._send(checks_to_html(checks, title="Environment check"))
            return
        self._send(form_html(text=default_doc()))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        ctype = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)

        text = ""
        goal = ""
        filename = ""
        if ctype.startswith("multipart/form-data"):
            m = re.search(r"boundary=([^;]+)", ctype)
            boundary = m.group(1).strip().strip('"').encode() if m else b""
            fields = _parse_multipart(body, boundary)
            text = fields.get("text", b"").decode("utf-8", "replace")
            goal = fields.get("goal", b"").decode("utf-8", "replace").strip()
            file_bytes = fields.get("file", b"")
            fname_field = b""
            for segment in body.split(b"--" + boundary):
                if b'name="file"' in segment:
                    fm = re.search(rb'filename="([^"]*)"', segment)
                    if fm:
                        fname_field = fm.group(1)
                    break
            filename = fname_field.decode("utf-8", "replace")

            if filename:
                try:
                    text = extract_text(filename, file_bytes)
                except DocumentError as e:
                    self._send(
                        form_html(goal=goal, error=render_error(L(f"文档解析失败：{html.escape(str(e))}",
                                                          f"Failed to parse the document: {html.escape(str(e))}"))),
                        code=400,
                    )
                    return
                if not text.strip():
                    text = ""
        else:
            form = parse_qs(body.decode("utf-8"))
            text = form.get("text", [""])[0]
            goal = form.get("goal", [""])[0].strip()

        if not text.strip():
            self._send(
                form_html(goal=goal, error=render_error(L("请输入或导入一段文本", "Paste or import some text first"))),
                code=400,
            )
            return
        source = filename or "粘贴文本"
        self._stream_result(text, goal, source)

    def _stream_result(self, text: str, goal: str, source: str) -> None:
        """Score with live progress: chunked HTML, small inline scripts."""
        assert SCORER is not None
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        title_html = html.escape(source) if source else L("阅读结果", "Reading result")
        sub = L(f"关注点：{html.escape(goal or '理解全文')}",
                f"Focus: {html.escape(goal or 'understanding the text')}")
        self._write_chunk(page_shell(title_html, doc_title=source, sub_html=sub))
        self._write_chunk(
            '<div class="progress" id="prog">'
            '<span id="prog-text">Splitting and scoring, keep this page open</span>'
            '<div class="track"><div class="bar" id="prog-bar"></div></div></div>'
        )
        t0 = time.time()

        def progress(done: int, total: int) -> None:
            elapsed = time.time() - t0
            rate = elapsed / max(done, 1)
            remain = int(rate * (total - done))
            pct = min(100, int(done * 100 / max(total, 1)))
            zh = json.dumps(f"已分析 {done} / {total} 句，预计剩余 {remain} 秒", ensure_ascii=False)
            en = json.dumps(f"Analyzed {done} / {total}, about {remain}s left", ensure_ascii=False)
            self._write_chunk(
                "<script>(function(){var s=document.documentElement.dataset.lang==='zh'?"
                + zh + ":" + en
                + ";document.getElementById('prog-text').textContent=s;"
                f"document.getElementById('prog-bar').style.width='{pct}%';}})();</script>"
            )

        sentences, total = SCORER.score(
            text, goal=goal or "understanding this text",
            progress=progress, max_sentences=SCORER.max_sentences(),
        )
        self._write_chunk("<script>document.getElementById('prog').style.display='none';</script>")
        self._write_chunk(
            result_body(sentences, goal=goal, source=source,
                        analyzed=min(total, SCORER.max_sentences()))
        )
        self._write_chunk(PAGE_FOOTER)
        self._write_chunk("", last=True)

    def _write_chunk(self, body: str, last: bool = False) -> None:
        if last:
            self.wfile.write(b"0\r\n\r\n")
        else:
            data = body.encode("utf-8")
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        pass


def main() -> None:
    global SCORER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    print("[reader] loading model ...", flush=True)
    SCORER = ImportanceScorer(model_dir=args.model_dir)
    print(f"[reader] backend={SCORER.backend} ready: http://127.0.0.1:{args.port}", flush=True)
    # MLX streams are thread-local -> keep the server single-threaded
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
