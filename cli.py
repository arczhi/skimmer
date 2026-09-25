"""CLI: score a text file and render importance highlights.

Usage:
    python cli.py article.txt --goal "finding key numbers" --html out.html
    python cli.py article.txt --terminal
    cat article.txt | python cli.py - --terminal --goal "action items"
"""

from __future__ import annotations

import argparse
import sys

from reader.render import to_html, to_terminal
from reader.scoring import ImportanceScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="text/markdown file or - for stdin")
    ap.add_argument("--goal", default="understanding this text",
                    help="user focus, e.g. 'finding key numbers' / 'action items'")
    ap.add_argument("--html", default=None, help="write colorized HTML here")
    ap.add_argument("--terminal", action="store_true", help="print ANSI-colored text")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    text = sys.stdin.read() if args.path == "-" else open(args.path, encoding="utf-8").read()
    scorer = ImportanceScorer(model_dir=args.model_dir)
    sentences = scorer.score(text, goal=args.goal)

    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(to_html(sentences, title=args.path, goal=args.goal))
        print(f"wrote {args.html} ({len(sentences)} sentences)")
    if args.terminal or not args.html:
        print(to_terminal(sentences))


if __name__ == "__main__":
    main()
