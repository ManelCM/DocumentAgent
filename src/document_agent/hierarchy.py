from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

from .types import BlockType, BoundingBox, DocumentBlock


# ──────────────────────────────────────────────────────────────────────────────
# Geometric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iou(a: DocumentBlock, b: DocumentBlock) -> float:
    ix1 = max(a.bbox.x1, b.bbox.x1)
    iy1 = max(a.bbox.y1, b.bbox.y1)
    ix2 = min(a.bbox.x2, b.bbox.x2)
    iy2 = min(a.bbox.y2, b.bbox.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.bbox.area + b.bbox.area - inter
    return inter / max(1, union)


def _overlap_ratio(child: DocumentBlock, parent: DocumentBlock) -> float:
    """Fraction of child's area that is covered by parent."""
    ix1 = max(child.bbox.x1, parent.bbox.x1)
    iy1 = max(child.bbox.y1, parent.bbox.y1)
    ix2 = min(child.bbox.x2, parent.bbox.x2)
    iy2 = min(child.bbox.y2, parent.bbox.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1, child.bbox.area)


def _same_line(a: DocumentBlock, b: DocumentBlock, tolerance: float = 0.5) -> bool:
    """
    True if two blocks sit on the same baseline / text line.
    Uses the smaller block's height as the tolerance unit.
    """
    min_h = min(a.bbox.height, b.bbox.height)
    if min_h == 0:
        return False
    return abs(a.bbox.cy - b.bbox.cy) <= min_h * tolerance


def _x_adjacent(a: DocumentBlock, b: DocumentBlock, gap_ratio: float = 1.5) -> bool:
    """
    True if the horizontal gap between a and b is small relative to their heights,
    i.e. they are side-by-side on the same line with no large horizontal separation.
    """
    gap = max(0, max(a.bbox.x1, b.bbox.x1) - min(a.bbox.x2, b.bbox.x2))
    ref = max(1, min(a.bbox.height, b.bbox.height))
    return gap <= ref * gap_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Containment-based parent detection (for figures that genuinely contain captions)
# ──────────────────────────────────────────────────────────────────────────────

def _find_container(
    block: DocumentBlock, candidates: List[DocumentBlock], min_overlap: float = 0.80
) -> Optional[DocumentBlock]:
    """Return the tightest block that contains ≥ min_overlap of block's area."""
    containers = [
        p for p in candidates
        if p.block_id != block.block_id
        and p.bbox.area > block.bbox.area
        and _overlap_ratio(block, p) >= min_overlap
    ]
    return min(containers, key=lambda c: c.bbox.area) if containers else None


# ──────────────────────────────────────────────────────────────────────────────
# Inline math stitching
# ──────────────────────────────────────────────────────────────────────────────

def _stitch_inline_math(page_blocks: List[DocumentBlock]) -> List[DocumentBlock]:
    """
    Detect formula blocks that are on the same line as a text block and
    create a MIXED parent block that covers both.

    The original text and formula blocks are marked skip_specialist=True;
    the MIXED block is dispatched to the text+formula specialist instead.
    """
    text_blocks = [b for b in page_blocks if b.block_type == BlockType.TEXT and not b.skip_specialist]
    formula_blocks = [b for b in page_blocks if b.block_type == BlockType.FORMULA and not b.skip_specialist]

    if not text_blocks or not formula_blocks:
        return page_blocks

    new_blocks: List[DocumentBlock] = []

    for formula in formula_blocks:
        # Find all text blocks on the same line and x-adjacent
        partners = [
            t for t in text_blocks
            if _same_line(formula, t) and _x_adjacent(formula, t)
        ]
        if not partners:
            continue

        # Include all partners + the formula in the MIXED group
        group = partners + [formula]

        # Merged bounding box
        mx1 = min(b.bbox.x1 for b in group)
        my1 = min(b.bbox.y1 for b in group)
        mx2 = max(b.bbox.x2 for b in group)
        my2 = max(b.bbox.y2 for b in group)

        mixed_id = f"mixed_{uuid.uuid4().hex[:8]}"
        mixed = DocumentBlock(
            block_id=mixed_id,
            page_index=formula.page_index,
            bbox=BoundingBox(mx1, my1, mx2, my2),
            block_type=BlockType.MIXED,
            detector_label="stitched_inline_math",
            confidence=min(b.confidence for b in group),
            reading_order=formula.reading_order,
            child_ids=[b.block_id for b in group],
        )
        # Tag the children so specialist nodes skip them
        for b in group:
            b.skip_specialist = True
            b.parent_id = mixed_id
            b.line_group_id = mixed_id

        new_blocks.append(mixed)

    return page_blocks + new_blocks


# ──────────────────────────────────────────────────────────────────────────────
# Caption → figure proximity linking
# ──────────────────────────────────────────────────────────────────────────────

_FIGURE_TYPES = {BlockType.FIGURE, BlockType.IMAGE, BlockType.CHART, BlockType.TABLE}


def _link_captions(page_blocks: List[DocumentBlock]) -> None:
    """Link each caption to the nearest figure/image/chart/table on the same page."""
    captions = [b for b in page_blocks if b.block_type == BlockType.CAPTION]
    targets = [b for b in page_blocks if b.block_type in _FIGURE_TYPES]

    for cap in captions:
        if not targets:
            break
        nearest = min(
            targets,
            key=lambda t: abs(t.bbox.cy - cap.bbox.cy) + abs(t.bbox.cx - cap.bbox.cx),
        )
        cap.relations["describes"] = nearest.block_id
        nearest.relations.setdefault("captions", []).append(cap.block_id)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_hierarchy(blocks: List[DocumentBlock]) -> List[DocumentBlock]:
    """
    Build block hierarchy using:
    1. Containment (≥80% overlap) for parent–child nesting.
    2. Same-line + x-adjacent detection for inline math → MIXED blocks.
    3. Caption proximity linking.
    """
    pages: Dict[int, List[DocumentBlock]] = {}
    for b in blocks:
        pages.setdefault(b.page_index, []).append(b)

    all_blocks: List[DocumentBlock] = []

    for page_blocks in pages.values():
        # 1. Containment-based parent assignment
        for b in page_blocks:
            candidates = [p for p in page_blocks if p.block_id != b.block_id]
            parent = _find_container(b, candidates, min_overlap=0.80)
            if parent:
                b.parent_id = parent.block_id
                if b.block_id not in parent.child_ids:
                    parent.child_ids.append(b.block_id)

        # 2. Inline math stitching → creates MIXED blocks
        page_blocks = _stitch_inline_math(page_blocks)

        # 3. Caption → figure linking
        _link_captions(page_blocks)

        all_blocks.extend(page_blocks)

    return all_blocks
