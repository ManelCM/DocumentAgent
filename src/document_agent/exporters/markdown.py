"""
Markdown exporter for DocumentAgent pipeline output.

Converts output.json → a clean, human-readable .md document that preserves:
  - Reading order across pages
  - Text paragraphs (de-hyphenated)
  - Section headings inferred from short all-caps / title-case text blocks
  - Formulas as LaTeX (inline $…$ or display $$…$$)
  - Tables as GFM markdown tables
  - Figures / images with caption and VLM summary
  - Charts with takeaway and inline data table
  - Captions in italic
  - Page markers (configurable)
  - Cross-reference links (block anchors)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Helpers ────────────────────────────────────────────────────────────────────

def _s(v: Any) -> str:
    """Safe string coerce; returns '' for None / non-string."""
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return ""
    return str(v).strip()


def _is_heading(text: str, block: Dict) -> bool:
    """Heuristic: short text blocks that look like section headings."""
    t = text.strip()
    if not t or len(t) > 120:
        return False
    # detector already labelled it title/heading
    label = (block.get("detector_label") or "").lower()
    if any(k in label for k in ("title", "heading", "section", "sub_title", "paragraph_title")):
        return True
    # Short line, ends without period, mostly title-case or all-caps
    if len(t) < 80 and not t.endswith("."):
        words = t.split()
        cap_words = sum(1 for w in words if w and (w[0].isupper() or w.isupper()))
        if cap_words / max(1, len(words)) >= 0.7:
            return True
    return False


def _md_table(headers: List[str], rows: List[Dict]) -> str:
    """Render a GFM markdown table."""
    if not headers:
        return ""
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("|" + "|".join(" :--- " for _ in headers) + "|")
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h)
            if v is None:
                cells.append("")
            else:
                cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _anchor(block_id: str) -> str:
    return f'<a id="{block_id}"></a>'


# ── Block renderers ────────────────────────────────────────────────────────────

def _render_text(block: Dict, block_map: Dict) -> str:
    p = block.get("payload", {})
    text = _s(p.get("text"))
    if not text:
        return ""

    lines = []
    if _is_heading(text, block):
        lines.append(f"\n## {text}\n")
    else:
        lines.append(text)

    # Embedded formula children
    for fid in block.get("relations", {}).get("embedded_formulas", []):
        child = block_map.get(fid)
        if not child:
            continue
        latex = _s(child.get("payload", {}).get("latex"))
        meaning = _s(child.get("payload", {}).get("meaning"))
        if latex:
            lines.append(f"\n$$\n{latex}\n$$")
            if meaning:
                lines.append(f"*{meaning}*")

    return "\n".join(lines)


def _render_mixed(block: Dict) -> str:
    p = block.get("payload", {})
    return _s(p.get("full_text") or p.get("plain_text") or p.get("text"))


def _render_formula(block: Dict) -> str:
    p = block.get("payload", {})
    latex   = _s(p.get("latex"))
    meaning = _s(p.get("meaning"))
    ftype   = _s(p.get("formula_type"))
    if not latex:
        return "*[formula not extracted]*"
    label = f"*{ftype}: {meaning}*" if ftype and meaning else f"*{meaning}*" if meaning else ""
    return f"\n$$\n{latex}\n$$\n{label}" if label else f"\n$$\n{latex}\n$$"


def _render_table(block: Dict, block_map: Dict) -> str:
    p = block.get("payload", {})
    title   = _s(p.get("title"))
    summary = _s(p.get("summary"))
    headers = p.get("headers") or []
    rows    = p.get("rows") or []
    notes   = _s(p.get("notes"))

    lines = []
    heading = title or summary
    if heading:
        lines.append(f"\n**Table: {heading}**\n")

    if headers and rows:
        lines.append(_md_table(headers, rows))
    elif summary:
        lines.append(f"*{summary}*")

    if notes:
        lines.append(f"\n_{notes}_")

    # Embedded formulas inside table cells
    for fid in block.get("relations", {}).get("embedded_formulas", []):
        child = block_map.get(fid)
        if not child:
            continue
        latex = _s(child.get("payload", {}).get("latex"))
        if latex:
            lines.append(f"\n$$\n{latex}\n$$")

    return "\n".join(lines)


def _render_image(block: Dict) -> str:
    p = block.get("payload", {})
    summary      = _s(p.get("summary"))
    visible_text = _s(p.get("visible_text"))
    key_elements = p.get("key_elements") or []

    lines = [f"\n> **[Figure]** {summary}" if summary else "\n> **[Figure]**"]
    if visible_text:
        lines.append(f"> *Visible text:* {visible_text}")
    if key_elements:
        lines.append(f"> *Elements:* {', '.join(str(e) for e in key_elements[:6])}")
    return "\n".join(lines)


def _render_chart(block: Dict) -> str:
    p = block.get("payload", {})
    chart_type = _s(p.get("chart_type"))
    title      = _s(p.get("title"))
    takeaway   = _s(p.get("takeaway"))
    trends     = _s(p.get("trends"))
    data_table = _s(p.get("data_table"))

    header = f"**[Chart"
    if chart_type:
        header += f" — {chart_type}"
    if title:
        header += f": {title}"
    header += "]**"

    lines = [f"\n> {header}"]
    if takeaway:
        lines.append(f"> *Insight:* {takeaway}")
    if trends:
        lines.append(f"> *Trends:* {trends}")
    if data_table:
        lines.append(f"\n{data_table}")
    return "\n".join(lines)


def _render_caption(block: Dict, block_map: Dict) -> str:
    p = block.get("payload", {})
    text = _s(p.get("text"))
    lines = []
    if text:
        lines.append(f"*{text}*")
    for fid in block.get("relations", {}).get("embedded_formulas", []):
        child = block_map.get(fid)
        if not child:
            continue
        latex = _s(child.get("payload", {}).get("latex"))
        if latex:
            lines.append(f"$$\n{latex}\n$$")
    return "\n".join(lines)


# ── Main exporter ──────────────────────────────────────────────────────────────

def to_markdown(
    output: Dict[str, Any],
    include_page_markers: bool = True,
    include_anchors: bool = True,
) -> str:
    """
    Convert a DocumentAgent output dict to a Markdown string.

    Parameters
    ----------
    output               : the dict from output.json
    include_page_markers : insert `---\\n*Page N*` between pages
    include_anchors      : insert <a id="bXX"> anchors for cross-linking
    """
    blocks = output.get("blocks", [])
    block_map: Dict[str, Dict] = {b["id"]: b for b in blocks}

    # Sort by reading order
    ordered = sorted(
        blocks,
        key=lambda b: (
            b.get("page_index", 0),
            b.get("reading_order", 999_999) if b.get("reading_order", -1) >= 0 else 999_999,
        ),
    )

    skip_types   = {"header", "footer", "other"}
    skip_labels  = {"aside_text"}
    skip_parents = {b["id"] for b in blocks
                    if b.get("parent_id") and b.get("type") != "mixed"}

    lines: List[str] = []
    # Document title from metadata
    input_name = Path(output.get("input_path", "document")).stem
    lines.append(f"# {input_name}\n")

    current_page = -1

    for block in ordered:
        btype  = block.get("type", "other")
        label  = block.get("detector_label", "")
        bid    = block.get("id", "")
        parent = block.get("parent_id")

        # Skip boilerplate
        if btype in skip_types:
            continue
        if label in skip_labels:
            continue
        # Skip children absorbed into parents (except MIXED which renders itself)
        if bid in skip_parents:
            continue
        # Skip noise: tiny text
        if btype == "text":
            raw = _s(block.get("payload", {}).get("text"))
            if len(raw) < 3:
                continue

        # Page marker
        page = block.get("page_index", 0)
        if include_page_markers and page != current_page:
            if current_page >= 0:
                lines.append("\n---")
            lines.append(f"\n*Page {page + 1}*\n")
            current_page = page

        # Anchor
        anchor = _anchor(bid) if include_anchors else ""

        rendered = ""
        if btype == "text":
            rendered = _render_text(block, block_map)
        elif btype == "mixed":
            rendered = _render_mixed(block)
        elif btype == "formula":
            rendered = _render_formula(block)
        elif btype == "table":
            rendered = _render_table(block, block_map)
        elif btype in ("image", "figure"):
            rendered = _render_image(block)
        elif btype == "chart":
            rendered = _render_chart(block)
        elif btype == "caption":
            rendered = _render_caption(block, block_map)

        if rendered:
            lines.append(f"{anchor}\n{rendered}" if anchor else rendered)

    return "\n\n".join(line for line in lines if line.strip())


def export_markdown(
    output: Dict[str, Any],
    output_path: str | Path,
    include_page_markers: bool = True,
    include_anchors: bool = True,
) -> Path:
    """Write the markdown string to *output_path* and return the Path."""
    md = to_markdown(output, include_page_markers, include_anchors)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    return p
