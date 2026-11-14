"""Build reports/execution_report.pdf — a submission PDF that embeds an
execution screenshot of every pipeline stage, the Airflow UI, the results, and
the GitHub repository link.

Pipeline:
  1. parse a captured full-run log (/tmp/exec_run.log) into per-stage segments
  2. render each stage's real console output as a terminal-styled PNG
     (headless Chrome screenshot)
  3. assemble a single PDF (xhtml2pdf) with those screenshots + Airflow UI
     screenshots + evaluation results table + repo link

Run (after a fresh `python pipeline.py 2>&1 | ... > /tmp/exec_run.log`):
    python scripts/build_execution_report.py
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LOG = Path("/tmp/exec_run.log")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO_URL = "https://github.com/cobra-kai-03/recommendation-pipeline"

STAGES = [
    ("ingestion", "Stage 1 — Data Ingestion", "python -m src.ingestion.ingest"),
    ("validation", "Stage 2 — Data Validation", "python -m src.validation.validate"),
    ("preparation", "Stage 3 — Data Preparation", "python -m src.preparation.prepare"),
    ("transformation", "Stage 4 — Data Transformation", "python -m src.transformation.transform"),
    ("features", "Stage 5 — Feature Management", "python -m src.features.build_features"),
    ("training", "Stage 6 — Model Training", "python -m src.training.train"),
    ("evaluation", "Stage 7 — Model Evaluation", "python -m src.evaluation.evaluate"),
]


def parse_segments(log_text: str) -> dict:
    """Return {stage_name: [log lines]} using the orchestrator markers."""
    segments, current, buf = {}, None, []
    for line in log_text.splitlines():
        m_start = re.search(r"STAGE START:\s*(\w+)", line)
        m_ok = re.search(r"STAGE OK:\s*(\w+)", line)
        if m_start:
            current = m_start.group(1)
            buf = [line]
            continue
        if m_ok and current:
            buf.append(line)
            segments[current] = buf
            current = None
            continue
        if current:
            buf.append(line)
    return segments


def render_terminal_png(cmd: str, lines: list, out_png: Path) -> None:
    body_lines = [f'<span class="cmd">$ {html.escape(cmd)}</span>']
    for ln in lines:
        body_lines.append(html.escape(ln))
    content = "\n".join(body_lines)
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      html,body {{ margin:0; padding:0; background:#1e1e24; }}
      .term {{ font-family:'SF Mono',Menlo,Consolas,monospace; font-size:14px;
               line-height:1.5; color:#d6d6dd; background:#1e1e24;
               padding:18px 22px; white-space:pre-wrap; }}
      .cmd {{ color:#4ec9b0; font-weight:bold; }}
      .bar {{ background:#2d2d35; padding:8px 14px; color:#9a9aa5;
              font-family:monospace; font-size:12px; }}
      .dot {{ height:11px; width:11px; border-radius:50%; display:inline-block;
              margin-right:6px; }}
    </style></head><body>
      <div class="bar">
        <span class="dot" style="background:#ff5f56"></span>
        <span class="dot" style="background:#ffbd2e"></span>
        <span class="dot" style="background:#27c93f"></span>
        &nbsp; Terminal — recommendation-pipeline
      </div>
      <div class="term">{content}</div>
    </body></html>"""

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(html_doc)
        tmp = fh.name
    height = 120 + len(body_lines) * 24
    subprocess.run([
        CHROME, "--headless=new", "--no-sandbox", "--disable-gpu",
        "--hide-scrollbars", "--force-device-scale-factor=2",
        f"--window-size=1200,{height}", f"--screenshot={out_png}",
        f"file://{tmp}",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def eval_table_html() -> str:
    import csv
    path = REPORTS / "model_evaluation.csv"
    if not path.exists():
        return ""
    with open(path) as fh:
        rows = list(csv.reader(fh))
    head, *body = rows
    def fmt(v):
        try:
            return f"{float(v):.4f}"
        except (ValueError, TypeError):
            return v or "—"
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    trs = ""
    for r in body:
        tds = "".join(f"<td>{html.escape(fmt(c))}</td>" for c in r)
        trs += f"<tr>{tds}</tr>"
    return f"<table class='data'><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def run_summary_html() -> str:
    path = REPORTS / "pipeline_run.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    rows = "".join(
        f"<tr><td>{html.escape(s['stage'])}</td><td>{s['status']}</td>"
        f"<td>{s['seconds']}s</td></tr>" for s in data["stages"])
    return (f"<p><b>Overall status:</b> {data['overall_status']} &nbsp;|&nbsp; "
            f"<b>Total time:</b> {data['total_seconds']}s</p>"
            f"<table class='data'><thead><tr><th>Stage</th><th>Status</th>"
            f"<th>Duration</th></tr></thead><tbody>{rows}</tbody></table>")


def main() -> None:
    import markdown  # noqa: F401  (ensures dep present)
    from xhtml2pdf import pisa

    segments = parse_segments(LOG.read_text())

    # 1. Render a terminal screenshot per stage.
    stage_imgs = {}
    for name, title, cmd in STAGES:
        lines = segments.get(name, [f"(no captured output for {name})"])
        out = REPORTS / f"exec_{name}.png"
        render_terminal_png(cmd, lines, out)
        stage_imgs[name] = out
        print("rendered", out.name)

    # 2. Compose the HTML document.
    css = """
    @page { size: A4; margin: 1.5cm; }
    body { font-family: Helvetica, Arial, sans-serif; color:#1b1b1b; font-size:11px; }
    h1 { color:#0b2e59; font-size:22px; margin:0 0 4px 0; }
    h2 { color:#0b2e59; font-size:15px; border-bottom:2px solid #0b2e59;
         padding-bottom:3px; margin-top:18px; }
    .meta { font-size:12px; line-height:1.6; }
    .meta a { color:#0b61d6; }
    img.shot { border:1px solid #ccc; margin:6px 0 2px 0; }
    .cap { color:#666; font-size:9.5px; margin:0 0 6px 2px; }
    table.data { border-collapse:collapse; width:100%; margin:8px 0; }
    table.data th, table.data td { border:1px solid #bbb; padding:4px 6px;
        font-size:9.5px; text-align:left; }
    table.data th { background:#0b2e59; color:#fff; }
    """

    parts = [f"<html><head><style>{css}</style></head><body>"]
    parts.append("<h1>Execution Report</h1>")
    parts.append(
        "<div class='meta'>"
        "<b>Title:</b> End-to-End Data Management Pipeline for a Recommendation System<br/>"
        "<b>Student:</b> Gouthami M &nbsp;|&nbsp; <b>ID:</b> 2025AG05395<br/>"
        "<b>Course:</b> Data Management for Machine Learning (DM4ML) — Assignment 2<br/>"
        "<b>Dataset:</b> MovieLens 100K (943 users, 1,682 items, 100,000 ratings)<br/>"
        f"<b>GitHub repository:</b> <a href='{REPO_URL}'>{REPO_URL}</a>"
        "</div>")

    parts.append("<h2>Pipeline run summary</h2>")
    parts.append(run_summary_html())

    # Per-stage execution screenshots.
    for name, title, cmd in STAGES:
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(f"<img class='shot' src='{stage_imgs[name]}' style='width:510px'/>")
        parts.append(f"<p class='cap'>Console output captured during "
                     f"<code>python pipeline.py</code> (stage: {name}).</p>")

    # Stage 8 — Orchestration (Airflow UI).
    parts.append("<h2>Stage 8 — Orchestration (Apache Airflow)</h2>")
    for img, cap in [
        ("airflow_ui_graph.png", "Airflow Graph view — all 7 tasks succeeded."),
        ("airflow_ui_grid.png", "Airflow Grid view — DAG run details: Total success = 1, Total tasks = 7."),
    ]:
        p = REPORTS / img
        if p.exists():
            parts.append(f"<img class='shot' src='{p}' style='width:510px'/>")
            parts.append(f"<p class='cap'>{cap}</p>")

    # Results.
    parts.append("<h2>Model evaluation results (K=10)</h2>")
    parts.append(eval_table_html())
    parts.append("<p class='cap'>Both models compared on the held-out test set. "
                 "Matrix factorization additionally reports RMSE/MAE.</p>")

    parts.append("</body></html>")
    html_doc = "\n".join(parts)

    out_pdf = REPORTS / "execution_report.pdf"
    with open(out_pdf, "wb") as fh:
        result = pisa.CreatePDF(html_doc, dest=fh)
    if result.err:
        raise RuntimeError("PDF generation failed")
    print("Wrote", out_pdf)


if __name__ == "__main__":
    main()
