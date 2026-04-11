from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .types import BlockType, DocumentBlock

# ──────────────────────────────────────────────────────────────────────────────
# LayoutReader (microsoft/layoutreader) — seq2seq permutation model
# Trained on ReadingBank (~500 k samples of academic / multi-column documents).
# ──────────────────────────────────────────────────────────────────────────────

_LAYOUTREADER_MODEL: Optional[object] = None
_LAYOUTREADER_TOKENIZER: Optional[object] = None
_LAYOUTREADER_LOADED: bool = False


def _load_layoutreader():
    global _LAYOUTREADER_MODEL, _LAYOUTREADER_TOKENIZER, _LAYOUTREADER_LOADED
    if _LAYOUTREADER_LOADED:
        return _LAYOUTREADER_MODEL, _LAYOUTREADER_TOKENIZER
    _LAYOUTREADER_LOADED = True
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = os.getenv("LAYOUTREADER_MODEL", "microsoft/layoutreader")
        _LAYOUTREADER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _LAYOUTREADER_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _LAYOUTREADER_MODEL.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LAYOUTREADER_MODEL.to(device)
    except Exception:
        _LAYOUTREADER_MODEL = None
        _LAYOUTREADER_TOKENIZER = None
    return _LAYOUTREADER_MODEL, _LAYOUTREADER_TOKENIZER


def _normalize_boxes(blocks: List[DocumentBlock], page_w: int, page_h: int) -> List[List[int]]:
    boxes = []
    for b in blocks:
        x1 = max(0, min(1000, int(b.bbox.x1 / page_w * 1000)))
        y1 = max(0, min(1000, int(b.bbox.y1 / page_h * 1000)))
        x2 = max(0, min(1000, int(b.bbox.x2 / page_w * 1000)))
        y2 = max(0, min(1000, int(b.bbox.y2 / page_h * 1000)))
        boxes.append([x1, y1, x2, y2])
    return boxes


def _layoutreader_predict(blocks: List[DocumentBlock], page_w: int, page_h: int) -> Optional[List[int]]:
    model, tokenizer = _load_layoutreader()
    if model is None or tokenizer is None:
        return None
    try:
        import torch

        boxes = _normalize_boxes(blocks, page_w, page_h)
        inputs = tokenizer(
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=len(blocks) + 2,
                num_beams=1,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        order = [int(x) for x in decoded.split() if x.isdigit()]
        if sorted(order) == list(range(len(blocks))):
            return order
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Column detection via x-center gap analysis on NARROW blocks
#
# Academic papers often have wide header blocks (title, authors, abstract)
# that span 55-70% of the page and cover the column separator gap in a
# pixel-coverage approach.  We fix this by:
#   1. Only using narrow blocks (width < 50% of page) for analysis.
#   2. Excluding HEADER / FOOTER blocks (boilerplate at page edges).
#   3. Finding the largest consecutive gap in block x-centers that falls
#      in the central 30-70% horizontal range of the page.
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_TYPES = {BlockType.HEADER, BlockType.FOOTER}


def _detect_column_boundaries(blocks: List[DocumentBlock], page_w: int) -> List[int]:
    """Return column separator x-positions (midpoints of gaps in block-center distribution).

    Strategy
    --------
    * Keep only narrow, body blocks (width < 50 % of page, not HEADER/FOOTER).
    * Collect their x-center positions and sort them.
    * Find the largest gap between consecutive centers that falls in the
      horizontal range [30 %, 70 %] of the page.
    * A gap must be at least 5 % of the page width to count as a separator.
    * Supports up to 3 columns (takes the top-3 qualifying gaps if the page
      has more than one separator).
    """
    if not blocks or page_w <= 0:
        return []

    narrow = [
        b for b in blocks
        if b.block_type not in _SKIP_TYPES
        and (b.bbox.x2 - b.bbox.x1) < 0.50 * page_w
    ]
    if len(narrow) < 3:
        return []

    centers = sorted(b.bbox.cx for b in narrow)
    mid_lo = 0.30 * page_w
    mid_hi = 0.70 * page_w
    min_gap = 0.05 * page_w   # at least 5 % of page width

    candidate_gaps: List[Tuple[float, float]] = []   # (gap_size, midpoint)
    for i in range(len(centers) - 1):
        lo, hi = centers[i], centers[i + 1]
        gap = hi - lo
        # The gap must be wide enough AND straddle the central zone
        if gap >= min_gap and lo <= mid_hi and hi >= mid_lo:
            candidate_gaps.append((gap, (lo + hi) / 2.0))

    if not candidate_gaps:
        return []

    # Sort by gap size descending; keep at most 2 separators (3 columns)
    candidate_gaps.sort(reverse=True)
    separators = sorted(int(mid) for _, mid in candidate_gaps[:2])
    return separators


def _assign_column(block: DocumentBlock, separators: List[int]) -> int:
    cx = block.bbox.cx
    col = 0
    for sep in sorted(separators):
        if cx > sep:
            col += 1
        else:
            break
    return col


def _is_full_width(block: DocumentBlock, page_w: int, threshold: float = 0.60) -> bool:
    """True if the block spans more than threshold of the page width."""
    return (block.bbox.x2 - block.bbox.x1) / max(1, page_w) >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic reading order (column-aware)
# ──────────────────────────────────────────────────────────────────────────────

def _heuristic_order(blocks: List[DocumentBlock], page_w: int, page_h: int) -> List[int]:
    """Column-aware heuristic reading order.

    Pipeline
    --------
    1. HEADER blocks → top of reading order (sorted by y).
    2. Full-width blocks (≥ 60 % of page width) → sorted by y; they read
       before the columnar body because they are titles / abstracts.
    3. Narrow body blocks → sorted by (column_index, cy) so left column
       reads completely before right column starts.
    4. FOOTER blocks → end of reading order (sorted by y).
    """
    separators = _detect_column_boundaries(blocks, page_w)

    headers  = [(i, b) for i, b in enumerate(blocks) if b.block_type == BlockType.HEADER]
    footers  = [(i, b) for i, b in enumerate(blocks) if b.block_type == BlockType.FOOTER]
    body     = [(i, b) for i, b in enumerate(blocks)
                if b.block_type not in {BlockType.HEADER, BlockType.FOOTER}]

    # Split body into spanning blocks (titles / abstracts) and columnar blocks
    spanning  = [(i, b) for i, b in body if _is_full_width(b, page_w)]
    columnar  = [(i, b) for i, b in body if not _is_full_width(b, page_w)]

    headers.sort(key=lambda x: x[1].bbox.cy)
    footers.sort(key=lambda x: x[1].bbox.cy)
    spanning.sort(key=lambda x: x[1].bbox.cy)

    if separators:
        columnar.sort(key=lambda x: (
            _assign_column(x[1], separators),
            x[1].bbox.cy,
            x[1].bbox.cx,
        ))
    else:
        # Single-column page: simple top-to-bottom, left-to-right
        columnar.sort(key=lambda x: (x[1].bbox.cy, x[1].bbox.cx))

    order = (
        [i for i, _ in headers]
        + [i for i, _ in spanning]
        + [i for i, _ in columnar]
        + [i for i, _ in footers]
    )
    return order


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def apply_reading_order(
    blocks: List[DocumentBlock],
    page_sizes: Optional[List[Dict]] = None,
) -> List[DocumentBlock]:
    """Assign reading_order index to every block.

    1. Tries LayoutReader (microsoft/layoutreader) first.
    2. Falls back to the column-aware heuristic.
    """
    by_page: Dict[int, List[DocumentBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_index, []).append(b)

    size_by_page: Dict[int, Dict] = {}
    if page_sizes:
        for ps in page_sizes:
            size_by_page[ps["page_index"]] = ps

    for page_idx, page_blocks in by_page.items():
        if not page_blocks:
            continue

        ps = size_by_page.get(page_idx, {})
        page_w = int(ps.get("width",  max(b.bbox.x2 for b in page_blocks)))
        page_h = int(ps.get("height", max(b.bbox.y2 for b in page_blocks)))

        order_indices = _layoutreader_predict(page_blocks, page_w, page_h)
        if order_indices is None:
            order_indices = _heuristic_order(page_blocks, page_w, page_h)

        for rank, idx in enumerate(order_indices):
            page_blocks[idx].reading_order = rank

    return blocks
