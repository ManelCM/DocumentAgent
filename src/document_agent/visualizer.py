"""
Visualizer — annotated page images + metrics charts.

Generates a report directory containing:
  report/
    pages/
      page_00.png   ← original page with coloured bbox overlays
      page_01.png
      ...
    charts/
      type_distribution.png
      engine_distribution.png
      node_timing.png
      blocks_per_page.png
    trace.json        ← full raw trace
    metrics.json      ← derived metrics summary
    report.html       ← self-contained HTML dashboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette per block type  (BGR for OpenCV)
# ──────────────────────────────────────────────────────────────────────────────

_COLOURS: Dict[str, tuple] = {
    "text":    (220, 220,  60),   # yellow
    "formula": ( 60, 180, 255),   # orange
    "image":   ( 60, 200,  60),   # green
    "figure":  ( 80, 220,  80),   # light green
    "chart":   (255, 100,  60),   # blue-ish
    "table":   (200,  60, 200),   # purple
    "caption": (180, 180, 180),   # grey
    "header":  (100, 200, 200),   # teal
    "footer":  (150, 150, 200),   # light purple
    "mixed":   (  0, 128, 255),   # amber
    "other":   (100, 100, 100),   # dark grey
}
_DEFAULT_COLOUR = (128, 128, 128)


def _colour(block_type: str) -> tuple:
    return _COLOURS.get(block_type, _DEFAULT_COLOUR)


# ──────────────────────────────────────────────────────────────────────────────
# Page annotation
# ──────────────────────────────────────────────────────────────────────────────

def annotate_page(
    page_img: np.ndarray,
    blocks: List[Dict],
    page_index: int,
) -> np.ndarray:
    """Draw coloured bounding boxes + reading order + type label on a page image."""
    img = page_img.copy()
    page_blocks = [b for b in blocks if b.get("page_index") == page_index]
    # Sort by reading order for label clarity
    page_blocks.sort(key=lambda b: b.get("reading_order", 999))

    for block in page_blocks:
        bbox = block.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1",0), bbox.get("y1",0), bbox.get("x2",0), bbox.get("y2",0)
        btype = block.get("type", "other")
        colour = _colour(btype)

        # Draw filled semi-transparent rectangle
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # Draw border
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # Label: reading_order + type (top-left corner of bbox)
        ro = block.get("reading_order", -1)
        label = f"#{ro} {btype}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        lx, ly = x1 + 2, max(y1 + th + 2, th + 4)
        # White background for legibility
        cv2.rectangle(img, (lx - 1, ly - th - 2), (lx + tw + 1, ly + 2), (255,255,255), -1)
        cv2.putText(img, label, (lx, ly), font, font_scale, colour, thickness, cv2.LINE_AA)

        # Engine badge (bottom-right corner)
        engine = block.get("engine", "")
        if engine:
            eng_short = engine[:8]
            (ew, eh), _ = cv2.getTextSize(eng_short, font, 0.35, 1)
            ex, ey = x2 - ew - 3, y2 - 3
            if ex > x1 and ey > y1:
                cv2.rectangle(img, (ex-1, ey-eh-1), (ex+ew+1, ey+2), (0,0,0), -1)
                cv2.putText(img, eng_short, (ex, ey), font, 0.35, (255,255,255), 1, cv2.LINE_AA)

    # Legend strip on the right side
    legend_x = img.shape[1] - 150
    if legend_x > 0:
        leg_y = 10
        for btype, col in _COLOURS.items():
            cv2.rectangle(img, (legend_x, leg_y), (legend_x + 18, leg_y + 14), col, -1)
            cv2.putText(img, btype, (legend_x + 22, leg_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30,30,30), 1, cv2.LINE_AA)
            leg_y += 18

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Metric charts (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def _save_bar(data: Dict[str, Any], title: str, xlabel: str, ylabel: str, path: Path, colour: str = "#4C72B0") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys = list(data.keys())
        vals = [data[k] for k in keys]

        fig, ax = plt.subplots(figsize=(max(6, len(keys) * 0.9), 4))
        bars = ax.bar(keys, vals, color=colour, edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%g", padding=2, fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Chart generation failed (%s): %s", title, exc)


def generate_charts(metrics: Dict, charts_dir: Path) -> Dict[str, str]:
    """Generate all metric charts. Returns {chart_name: relative_path}."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    td = metrics.get("type_distribution", {})
    if td:
        p = charts_dir / "type_distribution.png"
        colours = [f"#{abs(hash(k)) % 0xFFFFFF:06X}" for k in td]
        _save_bar(td, "Block Type Distribution", "Type", "Count", p, "#4C72B0")
        paths["type_distribution"] = str(p)

    ed = metrics.get("engine_distribution", {})
    if ed:
        p = charts_dir / "engine_distribution.png"
        _save_bar(ed, "OCR / VLM Engine Usage", "Engine", "Blocks processed", p, "#DD8452")
        paths["engine_distribution"] = str(p)

    nt = metrics.get("node_timing_s", {})
    if nt:
        p = charts_dir / "node_timing.png"
        _save_bar(nt, "Pipeline Node Timing", "Node", "Seconds", p, "#55A868")
        paths["node_timing"] = str(p)

    bp = metrics.get("blocks_per_page", {})
    if bp:
        p = charts_dir / "blocks_per_page.png"
        _save_bar({f"p{k}": v for k, v in sorted(bp.items())},
                  "Blocks per Page", "Page", "Blocks", p, "#C44E52")
        paths["blocks_per_page"] = str(p)

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────────────────

def _img_tag(rel_path: str, alt: str = "", width: str = "100%") -> str:
    return f'<img src="{rel_path}" alt="{alt}" style="max-width:{width};border-radius:6px;box-shadow:0 2px 8px #0002;">'


def generate_html(
    metrics: Dict,
    chart_paths: Dict[str, str],
    page_paths: List[str],
    report_dir: Path,
    input_path: str,
) -> Path:
    """Write report.html into report_dir."""
    html_path = report_dir / "report.html"

    def rel(p: str) -> str:
        return Path(p).relative_to(report_dir).as_posix()

    # Build block trace table rows
    trace_rows = ""
    for b in metrics.get("_blocks", []):
        preview = b.get("text_preview", "")[:80]
        trace_rows += (
            f"<tr>"
            f"<td>{b.get('block_id','')}</td>"
            f"<td>p{b.get('page_index','')}</td>"
            f"<td><span class='badge {b.get('type','')}'>{b.get('type','')}</span></td>"
            f"<td>{b.get('detector_label','')}</td>"
            f"<td>{b.get('confidence',''):.2f}</td>"
            f"<td>{b.get('reading_order','')}</td>"
            f"<td>{b.get('engine','')}</td>"
            f"<td>{b.get('duration_s',''):.3f}s</td>"
            f"<td title='{preview}'>{preview[:60]}{'…' if len(preview)>60 else ''}</td>"
            f"</tr>\n"
        )

    # Page image gallery
    pages_html = ""
    for i, pp in enumerate(page_paths):
        pages_html += f"<div class='page-card'><div class='page-label'>Page {i}</div>{_img_tag(rel(pp))}</div>\n"

    # Charts
    charts_html = ""
    for name, path in chart_paths.items():
        charts_html += f"<div class='chart-card'>{_img_tag(rel(path), name)}</div>\n"

    total_s = metrics.get("total_duration_s", 0) or 0
    num_blocks = metrics.get("num_blocks_traced", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Agent Report — {Path(input_path).name}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f5f6fa; color: #222; }}
  header {{ background: #1a1a2e; color: #fff; padding: 18px 32px; }}
  header h1 {{ margin:0; font-size:1.5rem; }}
  header p  {{ margin:4px 0 0; opacity:.7; font-size:.9rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  .stats {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:28px; }}
  .stat {{ background:#fff; border-radius:10px; padding:16px 24px; flex:1; min-width:140px;
           box-shadow:0 2px 8px #0001; text-align:center; }}
  .stat .val {{ font-size:2rem; font-weight:700; color:#4C72B0; }}
  .stat .lbl {{ font-size:.82rem; color:#666; margin-top:4px; }}
  h2 {{ font-size:1.15rem; margin:28px 0 12px; border-bottom:2px solid #e0e0e0; padding-bottom:6px; }}
  .charts {{ display:flex; flex-wrap:wrap; gap:16px; }}
  .chart-card {{ background:#fff; border-radius:10px; padding:12px; flex:1; min-width:280px;
                 box-shadow:0 2px 8px #0001; }}
  .pages {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .page-card {{ background:#fff; border-radius:10px; padding:10px; width:calc(50% - 14px);
                box-shadow:0 2px 8px #0001; }}
  .page-label {{ font-weight:600; margin-bottom:6px; font-size:.85rem; color:#555; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:10px;
           overflow:hidden; box-shadow:0 2px 8px #0001; font-size:.82rem; }}
  th {{ background:#1a1a2e; color:#fff; padding:8px 10px; text-align:left; }}
  td {{ padding:6px 10px; border-bottom:1px solid #f0f0f0; }}
  tr:hover td {{ background:#f8f9ff; }}
  .badge {{ display:inline-block; padding:1px 7px; border-radius:12px; font-size:.75rem;
            font-weight:600; color:#fff; background:#888; }}
  .badge.text    {{ background:#c8c83c; color:#333; }}
  .badge.formula {{ background:#3cb4ff; }}
  .badge.table   {{ background:#c83cc8; }}
  .badge.image,.badge.figure {{ background:#3cc83c; color:#333; }}
  .badge.chart   {{ background:#ff643c; }}
  .badge.mixed   {{ background:#ff8000; }}
  .badge.caption,.badge.header,.badge.footer {{ background:#aaa; color:#333; }}
</style>
</head>
<body>
<header>
  <h1>OCR Agent — Execution Report</h1>
  <p>{input_path}</p>
</header>
<div class="container">

  <div class="stats">
    <div class="stat"><div class="val">{total_s:.1f}s</div><div class="lbl">Total time</div></div>
    <div class="stat"><div class="val">{num_blocks}</div><div class="lbl">Blocks processed</div></div>
    <div class="stat"><div class="val">{len(page_paths)}</div><div class="lbl">Pages</div></div>
    <div class="stat"><div class="val">{len(metrics.get('type_distribution',{}))}</div><div class="lbl">Block types</div></div>
    <div class="stat"><div class="val">{len(metrics.get('engine_distribution',{}))}</div><div class="lbl">Engines used</div></div>
  </div>

  <h2>Metric Charts</h2>
  <div class="charts">{charts_html}</div>

  <h2>Annotated Pages</h2>
  <div class="pages">{pages_html}</div>

  <h2>Block Trace ({num_blocks} blocks)</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Page</th><th>Type</th><th>Label</th>
        <th>Conf</th><th>Order</th><th>Engine</th><th>Time</th><th>Preview</th>
      </tr>
    </thead>
    <tbody>{trace_rows}</tbody>
  </table>

</div>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    return html_path


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    output: Dict[str, Any],
    trace: Dict[str, Any],
    pages: List[np.ndarray],
    report_dir: Path,
) -> Path:
    """
    Full report generation pipeline.

    Parameters
    ----------
    output:   The agent output dict (from state["output"]).
    trace:    The raw trace dict (from state["_trace"]).
    pages:    List of page images (numpy BGR arrays).
    report_dir: Directory where report files are written.

    Returns
    -------
    Path to the generated report.html.
    """
    from .tracer import compute_metrics

    report_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = report_dir / "pages"
    charts_dir = report_dir / "charts"
    pages_dir.mkdir(exist_ok=True)

    blocks = output.get("blocks", [])
    input_path = output.get("input_path", "unknown")

    # ── 1. Annotate pages ────────────────────────────────────────────────────
    page_paths: List[str] = []
    for i, page_img in enumerate(pages):
        annotated = annotate_page(page_img, blocks, i)
        p = pages_dir / f"page_{i:02d}.png"
        cv2.imwrite(str(p), annotated)
        page_paths.append(str(p))
        logger.info("Annotated page %d → %s", i, p)

    # ── 2. Compute metrics ───────────────────────────────────────────────────
    metrics = compute_metrics(trace)
    # Attach block trace for HTML table
    metrics["_blocks"] = trace.get("blocks", [])

    # ── 3. Save raw trace + metrics ──────────────────────────────────────────
    (report_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    metrics_clean = {k: v for k, v in metrics.items() if k != "_blocks"}
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── 4. Charts ────────────────────────────────────────────────────────────
    chart_paths = generate_charts(metrics, charts_dir)

    # ── 5. HTML dashboard ────────────────────────────────────────────────────
    html_path = generate_html(metrics, chart_paths, page_paths, report_dir, input_path)
    logger.info("Report generated: %s", html_path)

    return html_path
