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
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from reader.diagnostics import checks_to_html, collect_checks
from reader.documents import DocumentError, extract_text
from reader.render import PAGE_CSS, PAGE_FOOTER, page_shell, result_body
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
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>重点阅读器</title>
<style>{css}
  .compose {{ display: grid; gap: 18px; margin-top: 26px; }}
  textarea {{ width: 100%; min-height: 380px; resize: vertical; padding: 16px 18px;
             border: 1px solid var(--border); border-radius: 10px; background: var(--surface);
             color: var(--ink); font: 400 15px/1.75 var(--sans); }}
  textarea:focus-visible, input[type=text]:focus-visible, input[type=file]:focus-visible {{
             outline: 2px solid var(--accent); outline-offset: 1px; border-color: transparent; }}
  .row {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
  .row label {{ font-size: 13px; color: var(--muted); min-width: 52px; }}
  input[type=text] {{ flex: 1; min-width: 220px; padding: 10px 12px; border: 1px solid var(--border);
             border-radius: 8px; background: var(--surface); color: var(--ink); font: 400 14px var(--sans); }}
  input[type=file] {{ font: 400 13px var(--sans); color: var(--muted); }}
  input[type=file]::file-selector-button {{ margin-right: 10px; padding: 8px 14px; border: 1px solid var(--border);
             border-radius: 8px; background: var(--surface); color: var(--ink); font: 500 13px var(--sans); cursor: pointer; }}
  input[type=file]::file-selector-button:hover {{ border-color: var(--accent); color: var(--accent-ink); }}
  button {{ justify-self: start; padding: 11px 26px; border: 0; border-radius: 8px;
            background: var(--accent); color: #fff; font: 500 15px var(--sans); cursor: pointer; }}
  button:hover {{ background: var(--accent-ink); }}
  .note {{ font-size: 13px; color: var(--muted); line-height: 1.7; }}
  .err {{ margin-top: 18px; padding: 12px 16px; border: 1px solid #e8c4c0; border-radius: 8px;
          background: #fdf2f1; color: #8f2f27; font-size: 14px; }}
</style>
</head>
<body>
<header class="top">
  <span class="mark">重点阅读器</span>
  <nav><a href="/plan">训练方案</a><a href="/status">环境自检</a></nav>
</header>
<main>
  <h1>粘贴文本，或导入文档</h1>
  <p class="lede">{lede}</p>
  {error}
  <form class="compose" method="post" action="/score" enctype="multipart/form-data">
    <textarea name="text" spellcheck="false" placeholder="粘贴要阅读的长文本。（导入文件时此处可留空）">{text}</textarea>
    <div class="row">
      <label for="file">文档</label>
      <input id="file" type="file" name="file" accept=".pdf,.docx,.doc,.txt,.md,.markdown">
      <span class="note">支持 PDF / DOCX / DOC / TXT / Markdown，解析在本机完成</span>
    </div>
    {goal_row}
    <button type="submit">标记重点</button>
  </form>
</main>
</body></html>"""


def render_error(msg: str) -> str:
    return f'<div class="err">{html.escape(msg)}</div>'


def lede_text() -> str:
    if SCORER is not None and SCORER.backend == "salience":
        return "模型在本地给每句话打重要性分，四档着色；整篇一次编码，纯 CPU 也秒级完成。"
    return "模型在本地给每句话打重要性分，四档着色。关注点可以随时改：同一篇文档换一个问题，重点就会变。"


def goal_row(goal: str) -> str:
    if SCORER is not None and SCORER.backend == "salience":
        return (
            '<div class="row"><label for="goal">关注点</label>'
            '<input id="goal" type="text" value="" disabled '
            'placeholder="当前模型按通用重要性打分">'
            '<span class="note">整篇模型暂为通用重要性，关注点条件化在后续版本</span></div>'
        )
    return (
        '<div class="row"><label for="goal">关注点</label>'
        f'<input id="goal" type="text" name="goal" value="{html.escape(goal)}" '
        'placeholder="可留空；例如：关键数字 / 行动项 / 主要论点 / 风险"></div>'
    )


def render_result(text: str, goal: str, source: str) -> str:
    assert SCORER is not None
    sentences, total = SCORER.score(
        text, goal=goal or "understanding this text",
        max_sentences=SCORER.max_sentences(),
    )
    return page_shell("重点标记") + result_body(
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
            checks = collect_checks(scorer=SCORER, smoke=SCORER is not None)
            self._send(checks_to_html(checks, title="环境自检"))
            return
        self._send(FORM.format(css=PAGE_CSS, text=html.escape(default_doc()), error="",
                            lede=lede_text(), goal_row=goal_row("")))

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
                        FORM.format(css=PAGE_CSS, text="", lede=lede_text(), goal_row=goal_row(goal),
                                    error=render_error(f"文档解析失败：{e}")),
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
                FORM.format(css=PAGE_CSS, text="", lede=lede_text(), goal_row=goal_row(goal),
                            error=render_error("请输入或导入一段文本")),
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
        self._write_chunk(page_shell("重点标记"))
        self._write_chunk(
            '<div class="progress" id="prog">正在分句并分析，请保持页面打开</div>'
        )
        t0 = time.time()

        def progress(done: int, total: int) -> None:
            elapsed = time.time() - t0
            rate = elapsed / max(done, 1)
            remain = int(rate * (total - done))
            self._write_chunk(
                "<script>document.getElementById('prog').textContent="
                f"'已分析 {done} / {total} 句，预计剩余 {remain} 秒';</script>"
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
