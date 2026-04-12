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

    # Opt-out: set LAYOUTREADER_ENABLED=false to skip the model entirely.
    # Default is false because the model takes ~6s to check on a cold run
    # when it is not locally cached.  Set LAYOUTREADER_ENABLED=true (or
    # point LAYOUTREADER_MODEL to a local path) to enable it.
    if os.getenv("LAYOUTREADER_ENABLED", "false").lower() not in {"1", "true", "yes"}:
        return None, None

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = os.getenv("LAYOUTREADER_MODEL", "microsoft/layoutreader")

        # Only attempt to load if the model is already cached locally.
        # This avoids a ~6 s network probe when the model is not available.
        try:
            from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
            weights_path = try_to_load_from_cache(model_name, "pytorch_model.bin")
            if weights_path is None or weights_path is _CACHED_NO_EXIST:
                # Model not cached — skip silently
                return None, None
        except Exception:
            pass  # huggingface_hub not available or API changed; try anyway

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
#   3. Finding the largest consecutive gap in block x-centers that straddles
#      the 20-80% horizontal range of the page.
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_TYPES = {BlockType.HEADER, BlockType.FOOTER}


def _detect_column_boundaries(blocks: List[DocumentBlock], page_w: int) -> List[int]:
    """Return column separator x-positions (midpoints of gaps in block-center distribution).

    Strategy
    --------
    * Keep only narrow, body blocks (width < 50 % of page, not HEADER/FOOTER).
    * Collect their x-center positions and sort them.
    * Find the largest gap between consecutive centers that straddles the
      horizontal range [20 %, 80 %] of the page.
    * A gap must be at least 5 % of the page width to count as a separator.
    * Supports up to 3 columns (takes the top-3 qualifying gaps if the page
      has more than one separator).
    """
    if not blocks or page_w <= 0:
        return []

    # Use blocks that are "column-sized": at least 4 % wide (filters margin noise)
    # and at most 50 % wide (filters full-width spanning blocks like titles).
    min_w = 0.04 * page_w
    narrow = [
        b for b in blocks
        if b.block_type not in _SKIP_TYPES
        and (b.bbox.x2 - b.bbox.x1) >= min_w
        and (b.bbox.x2 - b.bbox.x1) < 0.50 * page_w
    ]
    if len(narrow) < 2:
        return []

    centers = sorted(b.bbox.cx for b in narrow)
    # Separator search zone: the gap's *straddling range*.
    # [20%, 80%] allows asymmetric layouts (sidebars, narrow/wide columns,
    # 3-column where separators sit near 33 % and 67 %).
    # Keeping the zone away from the absolute page edges prevents false
    # positives caused by a lone margin annotation.
    mid_lo = 0.20 * page_w
    mid_hi = 0.80 * page_w
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

    # Sort by gap size descending so we always try the largest gaps first.
    candidate_gaps.sort(reverse=True)
    max_seps = min(2, len(candidate_gaps))   # support up to 3 columns

    # Validate: every resulting column must have ≥ N narrow blocks.
    # When falling back from n_seps to n_seps-1, always keep the LARGEST
    # gaps — do NOT just truncate the position-sorted list, which would
    # accidentally keep a small spurious gap over the true large gap.
    for n_seps in range(max_seps, 0, -1):
        # Take the n_seps LARGEST gaps, then sort them by x-position so
        # _assign_column (which iterates left-to-right) works correctly.
        seps = sorted(int(mid) for _, mid in candidate_gaps[:n_seps])
        # Build column boundaries [left_edge, sep0, sep1, ..., right_edge]
        boundaries = [0] + seps + [page_w]
        col_counts = [0] * (len(boundaries) - 1)
        for b in narrow:
            col = 0
            cx = b.bbox.cx
            for sep in seps:
                if cx > sep:
                    col += 1
                else:
                    break
            col_counts[col] += 1
        # Threshold scales with number of separators:
        #   2-column (1 sep): require ≥ 1 block per column.
        #     Gap position/size guards handle spurious detections; one-block
        #     columns occur on pages with a single reference block on one side.
        #   3-column (2 seps): require ≥ 4 blocks per column.
        #     Prevents 3 stray section-header blocks from creating a fake column.
        min_per_col = 1 if n_seps == 1 else 4
        if all(c >= min_per_col for c in col_counts):
            return seps

    return []


def _assign_column(block: DocumentBlock, separators: List[int]) -> int:
    cx = block.bbox.cx
    col = 0
    for sep in sorted(separators):
        if cx > sep:
            col += 1
        else:
            break
    return col


def _is_full_width(block: DocumentBlock, page_w: int, threshold: float = 0.50) -> bool:
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
    2. Full-width blocks (≥ 50 % of page width) → sorted by y; they read
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

    # Partition spanning blocks: those that begin ABOVE the first columnar block
    # are pre-column (titles, abstracts); those below are post-column (summary
    # figures, appendix-style full-width elements).  This ensures a full-width
    # figure that appears after the body columns doesn't displace column content.
    first_col_y = columnar[0][1].bbox.y1 if columnar else float("inf")
    pre_spanning  = [(i, b) for i, b in spanning if b.bbox.y1 <= first_col_y]
    post_spanning = [(i, b) for i, b in spanning if b.bbox.y1 > first_col_y]
    post_spanning.sort(key=lambda x: x[1].bbox.cy)

    order = (
        [i for i, _ in headers]
        + [i for i, _ in pre_spanning]
        + [i for i, _ in columnar]
        + [i for i, _ in post_spanning]
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

        # Only rank parent-level blocks; containment children inherit their
        # parent's context and do not occupy independent reading-order slots.
        parent_blocks = [b for b in page_blocks if b.parent_id is None]
        child_blocks  = [b for b in page_blocks if b.parent_id is not None]

        order_indices = _layoutreader_predict(parent_blocks, page_w, page_h)
        if order_indices is None:
            order_indices = _heuristic_order(parent_blocks, page_w, page_h)

        for rank, idx in enumerate(order_indices):
            parent_blocks[idx].reading_order = rank

        # Give children the same reading_order as their parent so they sort
        # adjacent to it (stable across downstream callers that sort by it).
        id_to_order = {b.block_id: b.reading_order for b in parent_blocks}
        for child in child_blocks:
            parent_ro = id_to_order.get(child.parent_id, -1)
            child.reading_order = parent_ro

    return blocks
