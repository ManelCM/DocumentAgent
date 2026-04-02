from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .types import BlockType, DocumentBlock

# ──────────────────────────────────────────────────────────────────────────────
# LayoutReader (microsoft/layoutreader) — deep learning reading order model
# trained on ReadingBank (~500 k samples of academic / multi-column documents).
# Normalises bounding boxes to [0, 1000] and runs a seq2seq permutation model.
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
    """Normalise bounding boxes to [0, 1000] as expected by LayoutReader."""
    boxes = []
    for b in blocks:
        x1 = max(0, min(1000, int(b.bbox.x1 / page_w * 1000)))
        y1 = max(0, min(1000, int(b.bbox.y1 / page_h * 1000)))
        x2 = max(0, min(1000, int(b.bbox.x2 / page_w * 1000)))
        y2 = max(0, min(1000, int(b.bbox.y2 / page_h * 1000)))
        boxes.append([x1, y1, x2, y2])
    return boxes


def _layoutreader_predict(blocks: List[DocumentBlock], page_w: int, page_h: int) -> Optional[List[int]]:
    """
    Run LayoutReader and return a list of block indices in reading order.
    Returns None if the model is unavailable or inference fails.
    """
    model, tokenizer = _load_layoutreader()
    if model is None or tokenizer is None:
        return None
    try:
        import torch

        boxes = _normalize_boxes(blocks, page_w, page_h)
        # LayoutReader tokenizer encodes boxes as special position tokens
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
        # Decode output token ids → block indices
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The model outputs space-separated indices e.g. "2 0 3 1"
        order = [int(x) for x in decoded.split() if x.isdigit()]
        # Validate: must be a permutation of [0, len(blocks)-1]
        if sorted(order) == list(range(len(blocks))):
            return order
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Improved heuristic fallback — column detection via x-projection gap analysis
# ──────────────────────────────────────────────────────────────────────────────

def _detect_column_boundaries(blocks: List[DocumentBlock], page_w: int) -> List[int]:
    """
    Find column separator x-positions by looking for vertical gaps
    (x-ranges not covered by any block).
    Returns sorted list of column separator x-positions (midpoints of gaps).
    """
    if not blocks or page_w <= 0:
        return []

    # Exclude full-width blocks (titles, abstracts) — they span the whole page
    # and would mask column gaps in the histogram.
    columnar = [b for b in blocks if not _is_full_width(b, page_w)]
    if not columnar:
        return []

    # Build coverage array
    covered = [0] * page_w
    for b in columnar:
        for x in range(max(0, b.bbox.x1), min(page_w, b.bbox.x2)):
            covered[x] += 1

    # Find gap segments (covered[x] == 0) in the central 20%-80% horizontal range
    # to avoid being fooled by margins
    left_margin = int(page_w * 0.20)
    right_margin = int(page_w * 0.80)
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for x in range(left_margin, right_margin):
        if covered[x] == 0:
            if not in_gap:
                in_gap = True
                gap_start = x
        else:
            if in_gap:
                in_gap = False
                gap_end = x
                if gap_end - gap_start >= int(page_w * 0.015):  # min 1.5% gap width
                    gaps.append((gap_start, gap_end))
    if in_gap:
        gaps.append((gap_start, right_margin))

    return [(s + e) // 2 for s, e in gaps]


def _assign_column(block: DocumentBlock, separators: List[int]) -> int:
    """Return 0-based column index for a block given column separator x-positions."""
    cx = block.bbox.cx
    col = 0
    for sep in sorted(separators):
        if cx > sep:
            col += 1
        else:
            break
    return col


def _is_full_width(block: DocumentBlock, page_w: int, threshold: float = 0.65) -> bool:
    """True if the block spans more than threshold of the page width (title, abstract, etc.)."""
    return (block.bbox.x2 - block.bbox.x1) / max(1, page_w) >= threshold


def _heuristic_order(
    blocks: List[DocumentBlock], page_w: int, page_h: int
) -> List[int]:
    """
    Improved column-aware heuristic:
    1. Full-width blocks (titles, abstracts) are sorted first by y-centre.
    2. Remaining blocks are assigned to columns and sorted column-by-column, top-to-bottom.
    Returns list of original indices in reading order.
    """
    separators = _detect_column_boundaries(blocks, page_w)

    full_width = [(i, b) for i, b in enumerate(blocks) if _is_full_width(b, page_w)]
    columnar = [(i, b) for i, b in enumerate(blocks) if not _is_full_width(b, page_w)]

    # Sort full-width blocks by y (top → bottom)
    full_width.sort(key=lambda x: x[1].bbox.cy)

    # For columnar blocks, separate HEADER/FOOTER (always first/last)
    headers = [(i, b) for i, b in columnar if b.block_type in {BlockType.HEADER}]
    footers = [(i, b) for i, b in columnar if b.block_type in {BlockType.FOOTER}]
    body = [(i, b) for i, b in columnar if b.block_type not in {BlockType.HEADER, BlockType.FOOTER}]

    headers.sort(key=lambda x: x[1].bbox.cy)
    footers.sort(key=lambda x: x[1].bbox.cy)

    if separators:
        body.sort(key=lambda x: (_assign_column(x[1], separators), x[1].bbox.cy, x[1].bbox.cx))
    else:
        body.sort(key=lambda x: (x[1].bbox.cy, x[1].bbox.cx))

    order = (
        [i for i, _ in headers]
        + [i for i, _ in full_width]
        + [i for i, _ in body]
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
    """
    Assign reading_order index to every block.

    Strategy (per page):
      1. Try LayoutReader (microsoft/layoutreader trained on ReadingBank).
      2. Fall back to improved column-aware heuristic.
    """
    by_page: Dict[int, List[DocumentBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_index, []).append(b)

    # Build page size lookup
    size_by_page: Dict[int, Dict] = {}
    if page_sizes:
        for ps in page_sizes:
            size_by_page[ps["page_index"]] = ps

    for page_idx, page_blocks in by_page.items():
        if not page_blocks:
            continue

        ps = size_by_page.get(page_idx, {})
        page_w = int(ps.get("width", max(b.bbox.x2 for b in page_blocks)))
        page_h = int(ps.get("height", max(b.bbox.y2 for b in page_blocks)))

        # Attempt LayoutReader
        order_indices = _layoutreader_predict(page_blocks, page_w, page_h)

        # Fallback
        if order_indices is None:
            order_indices = _heuristic_order(page_blocks, page_w, page_h)

        for rank, idx in enumerate(order_indices):
            page_blocks[idx].reading_order = rank

    return blocks
