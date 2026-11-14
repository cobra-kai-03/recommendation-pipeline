"""Export a Markdown report to PDF (pure-Python: markdown + xhtml2pdf).

Usage:
    python scripts/export_report_pdf.py reports/final_report.md
    python scripts/export_report_pdf.py            # defaults to final_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5px; color: #222; line-height: 1.45; }
h1 { font-size: 20px; color: #1a3d6d; border-bottom: 2px solid #1a3d6d; padding-bottom: 4px; }
h2 { font-size: 15px; color: #1a3d6d; margin-top: 16px; }
h3 { font-size: 12.5px; color: #333; }
code { background: #f2f2f2; padding: 1px 3px; font-family: Courier, monospace; font-size: 9.5px; }
pre { background: #f6f6f6; padding: 8px; font-size: 9px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 4px 6px; font-size: 9.5px; text-align: left; }
th { background: #1a3d6d; color: #fff; }
"""


def convert(md_path: Path) -> Path:
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"
    pdf_path = md_path.with_suffix(".pdf")
    with open(pdf_path, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh)
    if result.err:
        raise RuntimeError(f"PDF generation failed for {md_path}")
    return pdf_path


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "final_report.md"
    out = convert(target)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
