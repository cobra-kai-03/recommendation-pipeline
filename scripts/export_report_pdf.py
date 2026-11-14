"""Export a Markdown report to PDF (pure-Python: markdown + xhtml2pdf).

Usage:
    python scripts/export_report_pdf.py reports/final_report.md
    python scripts/export_report_pdf.py            # defaults to final_report.md
"""
from __future__ import annotations

import base64
import re
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
img { max-width: 100%; height: auto; margin: 8px 0; }
"""


def _embed_images(html: str, base_dir: Path) -> str:
    """Replace relative img src attributes with base64 data URIs."""
    def replace(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
            return m.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            return m.group(0)
        ext = img_path.suffix.lstrip(".").lower()
        mime = "image/png" if ext == "png" else f"image/{ext}"
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        return f'src="data:{mime};base64,{b64}"'
    return re.sub(r'src="([^"]+)"', replace, html)


def convert(md_path: Path) -> Path:
    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"
    html = _embed_images(html, md_path.parent)
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
