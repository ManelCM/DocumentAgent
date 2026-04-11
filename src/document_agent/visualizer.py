"""
Visualizer — annotated page images + metrics charts + HTML dashboard.

Report directory layout:
  report/
    pages/
      page_00.png   ← original page with coloured bbox overlays + reading order
    charts/
      type_distribution.png
      engine_distribution.png
      node_timing.png
      blocks_per_page.png
      token_usage.png
    trace.json        ← full raw trace (prompts, responses, tokens, timing)
    metrics.json      ← derived metrics summary
    full_text.txt     ← reading-order concatenated text for RAG
    report.html       ← self-contained HTML dashboard
"""

from __future__ import annotations

import html as _html
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
    "text":    (220, 220,  60),
    "formula": ( 60, 180, 255),
    "image":   ( 60, 200,  60),
    "figure":  ( 80, 220,  80),
    "chart":   (255, 100,  60),
    "table":   (200,  60, 200),
    "caption": (180, 180, 180),
    "header":  (100, 200, 200),
    "footer":  (150, 150, 200),
    "mixed":   (  0, 128, 255),
    "other":   (100, 100, 100),
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
    """Draw coloured bounding boxes + reading order number + type label."""
    img = page_img.copy()
    page_blocks = [b for b in blocks if b.get("page_index") == page_index]
    page_blocks.sort(key=lambda b: b.get("reading_order", 999))

    for block in page_blocks:
        # Skip child blocks (formula/text inside a parent block) — they are visual noise.
        # Children have parent_id set.  MIXED blocks are themselves parents, not children.
        if block.get("parent_id") is not None:
            continue

        bbox = block.get("bbox", {})
        x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
        x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
        btype  = block.get("type", "other")
        colour = _colour(btype)

        # Semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # Border
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

        # Label: reading_order + type
        ro    = block.get("reading_order", -1)
        label = f"#{ro} {btype}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.42
        (tw, th), _ = cv2.getTextSize(label, font, fscale, 1)
        lx = x1 + 2
        ly = max(y1 + th + 2, th + 4)
        cv2.rectangle(img, (lx - 1, ly - th - 2), (lx + tw + 1, ly + 2), (255, 255, 255), -1)
        cv2.putText(img, label, (lx, ly), font, fscale, colour, 1, cv2.LINE_AA)

        # Engine badge (bottom-right)
        engine = block.get("engine", "")
        if engine:
            eng_short = engine[:10]
            (ew, eh), _ = cv2.getTextSize(eng_short, font, 0.32, 1)
            ex, ey = x2 - ew - 3, y2 - 3
            if ex > x1 and ey > y1:
                cv2.rectangle(img, (ex - 1, ey - eh - 1), (ex + ew + 1, ey + 2), (30, 30, 30), -1)
                cv2.putText(img, eng_short, (ex, ey), font, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend strip
    legend_x = img.shape[1] - 155
    if legend_x > 0:
        leg_y = 10
        for btype, col in _COLOURS.items():
            cv2.rectangle(img, (legend_x, leg_y), (legend_x + 18, leg_y + 14), col, -1)
            cv2.putText(img, btype, (legend_x + 22, leg_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
            leg_y += 18

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Metric charts (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def _save_bar(data: Dict[str, Any], title: str, xlabel: str, ylabel: str,
              path: Path, colour: str = "#4C72B0") -> None:
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
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    if metrics.get("type_distribution"):
        p = charts_dir / "type_distribution.png"
        _save_bar(metrics["type_distribution"], "Block Type Distribution", "Type", "Count", p, "#4C72B0")
        paths["type_distribution"] = str(p)

    if metrics.get("engine_distribution"):
        p = charts_dir / "engine_distribution.png"
        _save_bar(metrics["engine_distribution"], "OCR / VLM Engine Usage", "Engine", "Blocks", p, "#DD8452")
        paths["engine_distribution"] = str(p)

    if metrics.get("node_timing_s"):
        p = charts_dir / "node_timing.png"
        _save_bar(metrics["node_timing_s"], "Pipeline Node Timing", "Node", "Seconds", p, "#55A868")
        paths["node_timing"] = str(p)

    if metrics.get("blocks_per_page"):
        p = charts_dir / "blocks_per_page.png"
        _save_bar({f"p{k}": v for k, v in sorted(metrics["blocks_per_page"].items())},
                  "Blocks per Page", "Page", "Blocks", p, "#C44E52")
        paths["blocks_per_page"] = str(p)

    # Token usage per model
    breakdown = metrics.get("model_breakdown", {})
    if breakdown:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            models = list(breakdown.keys())
            pt = [breakdown[m]["prompt_tokens"]     for m in models]
            ct = [breakdown[m]["completion_tokens"] for m in models]
            x  = range(len(models))

            fig, ax = plt.subplots(figsize=(max(5, len(models) * 1.5), 4))
            bars_p = ax.bar(x, pt, label="Prompt tokens",     color="#4C72B0")
            bars_c = ax.bar(x, ct, bottom=pt, label="Completion tokens", color="#DD8452")
            ax.set_xticks(list(x))
            ax.set_xticklabels(models, rotation=15)
            ax.set_title("Token Usage by Model", fontsize=13, fontweight="bold", pad=10)
            ax.set_ylabel("Tokens")
            ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            p = charts_dir / "token_usage.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths["token_usage"] = str(p)
        except Exception as exc:
            logger.warning("Token usage chart failed: %s", exc)

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────────────────

def _e(s: Any) -> str:
    """HTML-escape a value."""
    return _html.escape(str(s))


def _img_tag(rel_path: str, alt: str = "", width: str = "100%") -> str:
    return f'<img src="{rel_path}" alt="{_e(alt)}" style="max-width:{width};border-radius:6px;box-shadow:0 2px 8px #0002;">'


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


def generate_html(
    metrics: Dict,
    chart_paths: Dict[str, str],
    page_paths: List[str],
    report_dir: Path,
    input_path: str,
    full_text: str = "",
) -> Path:
    html_path = report_dir / "report.html"

    def rel(p: str) -> str:
        return Path(p).relative_to(report_dir).as_posix()

    # ── Block trace table ────────────────────────────────────────────────────
    trace_rows = ""
    for b in metrics.get("_blocks", []):
        preview = _e(b.get("text_preview", "")[:80])
        trace_rows += (
            f"<tr>"
            f"<td>{_e(b.get('block_id',''))}</td>"
            f"<td>p{_e(b.get('page_index',''))}</td>"
            f"<td><span class='badge {_e(b.get('type',''))}'>{_e(b.get('type',''))}</span></td>"
            f"<td>{_e(b.get('detector_label',''))}</td>"
            f"<td>{float(b.get('confidence', 0)):.2f}</td>"
            f"<td>{_e(b.get('reading_order',''))}</td>"
            f"<td>{_e(b.get('engine',''))}</td>"
            f"<td>{float(b.get('duration_s', 0)):.3f}s</td>"
            f"<td title='{preview}'>{preview[:60]}{'…' if len(preview) > 60 else ''}</td>"
            f"</tr>\n"
        )

    # ── LLM calls audit table ────────────────────────────────────────────────
    llm_rows = ""
    for c in metrics.get("_llm_calls", []):
        resp_preview = _e(str(c.get("response_text", ""))[:120])
        llm_rows += (
            f"<tr>"
            f"<td>{_e(c.get('block_id',''))}</td>"
            f"<td>p{_e(c.get('page_index',''))}</td>"
            f"<td><span class='badge {_e(c.get('block_type',''))}'>{_e(c.get('block_type',''))}</span></td>"
            f"<td>{_e(c.get('model',''))}</td>"
            f"<td><code>{_e(c.get('prompt_name',''))}</code></td>"
            f"<td class='num'>{_fmt_tokens(c.get('prompt_tokens', 0))}</td>"
            f"<td class='num'>{_fmt_tokens(c.get('completion_tokens', 0))}</td>"
            f"<td class='num'>{_fmt_tokens(c.get('total_tokens', 0))}</td>"
            f"<td class='num'>${c.get('cost_usd', 0):.5f}</td>"
            f"<td>{float(c.get('duration_s', 0)):.2f}s</td>"
            f"<td title='{resp_preview}'>{resp_preview[:80]}{'…' if len(resp_preview) > 80 else ''}</td>"
            f"</tr>\n"
        )

    # ── Page gallery ─────────────────────────────────────────────────────────
    pages_html = ""
    for i, pp in enumerate(page_paths):
        pages_html += f"<div class='page-card'><div class='page-label'>Page {i + 1}</div>{_img_tag(rel(pp))}</div>\n"

    # ── Metric charts ─────────────────────────────────────────────────────────
    charts_html = ""
    for name, path in chart_paths.items():
        charts_html += f"<div class='chart-card'>{_img_tag(rel(path), name)}</div>\n"

    # ── Summary numbers ───────────────────────────────────────────────────────
    total_s    = metrics.get("total_duration_s", 0) or 0
    num_blocks = metrics.get("num_blocks_traced", 0)
    num_llm    = metrics.get("num_llm_calls", 0)
    pt         = metrics.get("total_prompt_tokens", 0)
    ct         = metrics.get("total_completion_tokens", 0)
    tt         = metrics.get("total_tokens", 0)
    cost       = metrics.get("total_cost_usd", 0.0)

    # ── Full text (escape for HTML) ───────────────────────────────────────────
    ft_html = _e(full_text[:20000]) + ("…" if len(full_text) > 20000 else "")
    ft_html = ft_html.replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Agent Report — {_e(Path(input_path).name)}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin:0; background:#f5f6fa; color:#222; }}
  header {{ background:#1a1a2e; color:#fff; padding:18px 32px; }}
  header h1 {{ margin:0; font-size:1.5rem; }}
  header p  {{ margin:4px 0 0; opacity:.7; font-size:.9rem; }}
  .container {{ max-width:1500px; margin:0 auto; padding:24px; }}
  .stats {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:28px; }}
  .stat {{ background:#fff; border-radius:10px; padding:14px 20px; flex:1; min-width:120px;
           box-shadow:0 2px 8px #0001; text-align:center; }}
  .stat .val {{ font-size:1.7rem; font-weight:700; color:#4C72B0; }}
  .stat .lbl {{ font-size:.78rem; color:#666; margin-top:3px; }}
  h2 {{ font-size:1.1rem; margin:28px 0 10px; border-bottom:2px solid #e0e0e0; padding-bottom:5px; }}
  .charts {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .chart-card {{ background:#fff; border-radius:10px; padding:12px; flex:1; min-width:280px;
                 box-shadow:0 2px 8px #0001; }}
  .pages {{ display:flex; flex-wrap:wrap; gap:12px; }}
  .page-card {{ background:#fff; border-radius:10px; padding:10px; width:calc(50% - 14px);
                box-shadow:0 2px 8px #0001; }}
  .page-label {{ font-weight:600; margin-bottom:5px; font-size:.82rem; color:#555; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:10px;
           overflow:hidden; box-shadow:0 2px 8px #0001; font-size:.78rem; }}
  th {{ background:#1a1a2e; color:#fff; padding:7px 9px; text-align:left; white-space:nowrap; }}
  td {{ padding:5px 9px; border-bottom:1px solid #f0f0f0; word-break:break-word; max-width:300px; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr:hover td {{ background:#f8f9ff; }}
  .badge {{ display:inline-block; padding:1px 7px; border-radius:12px; font-size:.72rem;
            font-weight:600; color:#fff; background:#888; }}
  .badge.text    {{ background:#c8c83c; color:#333; }}
  .badge.formula {{ background:#3cb4ff; }}
  .badge.table   {{ background:#c83cc8; }}
  .badge.image,.badge.figure {{ background:#3cc83c; color:#333; }}
  .badge.chart   {{ background:#ff643c; }}
  .badge.mixed   {{ background:#ff8000; }}
  .badge.caption,.badge.header,.badge.footer {{ background:#aaa; color:#333; }}
  .full-text {{ background:#fff; border-radius:10px; padding:20px; box-shadow:0 2px 8px #0001;
               font-family:'Courier New',monospace; font-size:.78rem; line-height:1.6;
               max-height:600px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; }}
  code {{ background:#f0f0f4; padding:1px 5px; border-radius:4px; font-size:.85em; }}
  .cost {{ color:#2a7a2a; font-weight:600; }}
</style>
</head>
<body>
<header>
  <h1>OCR Agent — Execution Report</h1>
  <p>{_e(input_path)}</p>
</header>
<div class="container">

  <div class="stats">
    <div class="stat"><div class="val">{total_s:.1f}s</div><div class="lbl">Total time</div></div>
    <div class="stat"><div class="val">{num_blocks}</div><div class="lbl">Blocks processed</div></div>
    <div class="stat"><div class="val">{len(page_paths)}</div><div class="lbl">Pages</div></div>
    <div class="stat"><div class="val">{num_llm}</div><div class="lbl">LLM calls</div></div>
    <div class="stat"><div class="val">{_fmt_tokens(pt)}</div><div class="lbl">Prompt tokens</div></div>
    <div class="stat"><div class="val">{_fmt_tokens(ct)}</div><div class="lbl">Completion tokens</div></div>
    <div class="stat"><div class="val">{_fmt_tokens(tt)}</div><div class="lbl">Total tokens</div></div>
    <div class="stat"><div class="val cost">${cost:.4f}</div><div class="lbl">Estimated cost</div></div>
  </div>

  <h2>Metric Charts</h2>
  <div class="charts">{charts_html}</div>

  <h2>Annotated Pages</h2>
  <div class="pages">{pages_html}</div>

  <h2>Full Document Text ({len(full_text):,} chars) — Reading Order</h2>
  <div class="full-text">{ft_html}</div>

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

  <h2>LLM Call Audit ({num_llm} calls — ${cost:.4f} total)</h2>
  <table>
    <thead>
      <tr>
        <th>Block</th><th>Page</th><th>Type</th><th>Model</th><th>Prompt</th>
        <th>Prompt↑</th><th>Compl.↓</th><th>Total</th><th>Cost</th><th>Time</th><th>Response preview</th>
      </tr>
    </thead>
    <tbody>{llm_rows}</tbody>
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
    from .tracer import compute_metrics

    report_dir.mkdir(parents=True, exist_ok=True)
    pages_dir  = report_dir / "pages"
    charts_dir = report_dir / "charts"
    pages_dir.mkdir(exist_ok=True)

    blocks     = output.get("blocks", [])
    input_path = output.get("input_path", "unknown")
    full_text  = output.get("full_text", "")

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
    metrics["_blocks"]    = trace.get("blocks", [])
    metrics["_llm_calls"] = trace.get("llm_calls", [])

    # ── 3. Save raw trace + metrics + full text ──────────────────────────────
    (report_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    metrics_clean = {k: v for k, v in metrics.items() if not k.startswith("_")}
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_clean, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (report_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
    logger.info("Full text → %s (%d chars)", report_dir / "full_text.txt", len(full_text))

    # ── 4. Charts ────────────────────────────────────────────────────────────
    chart_paths = generate_charts(metrics, charts_dir)

    # ── 5. HTML dashboard ────────────────────────────────────────────────────
    html_path = generate_html(metrics, chart_paths, page_paths, report_dir, input_path, full_text)
    logger.info("Report generated: %s", html_path)

    return html_path
