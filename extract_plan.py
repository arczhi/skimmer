"""Extract readable plain text from plan.html for the reader default document."""

from html.parser import HTMLParser


class Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag in ("style", "script"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3"):
            self.parts.append("\n\n")
        elif tag in ("p", "tr", "pre"):
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "td":
            self.parts.append(" | ")

    def handle_endtag(self, tag) -> None:
        if tag in ("style", "script"):
            self.skip -= 1
        elif tag in ("h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_data(self, data) -> None:
        if self.skip:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [ln.rstrip() for ln in raw.split("\n")]
        out: list[str] = []
        for ln in lines:
            ln = ln.strip()
            if not ln or (out and out[-1] == ""):
                if ln or (out and out[-1] != ""):
                    out.append(ln)
                continue
            out.append(ln)
        return "\n".join(out).strip() + "\n"


if __name__ == "__main__":
    p = Extractor()
    p.feed(open("plan.html", encoding="utf-8").read())
    open("plan.txt", "w", encoding="utf-8").write(p.text())
    print("wrote plan.txt,", len(p.text()), "chars")
