from __future__ import annotations

from typing import List, Optional

from .types import BlockType, DocumentBlock


def _contains(parent: DocumentBlock, child: DocumentBlock, overlap_ratio: float = 0.8) -> bool:
    px1, py1, px2, py2 = parent.bbox.x1, parent.bbox.y1, parent.bbox.x2, parent.bbox.y2
    cx1, cy1, cx2, cy2 = child.bbox.x1, child.bbox.y1, child.bbox.x2, child.bbox.y2
    ix1, iy1 = max(px1, cx1), max(py1, cy1)
    ix2, iy2 = min(px2, cx2), min(py2, cy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter_area = iw * ih
    child_area = max(1, child.bbox.area)
    return (inter_area / child_area) >= overlap_ratio


def _find_parent(block: DocumentBlock, page_blocks: List[DocumentBlock]) -> Optional[DocumentBlock]:
    candidates = [
        p
        for p in page_blocks
        if p.block_id != block.block_id and p.bbox.area > block.bbox.area and _contains(p, block, overlap_ratio=0.9)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.bbox.area)


def build_hierarchy(blocks: List[DocumentBlock]) -> List[DocumentBlock]:
    pages = {}
    for b in blocks:
        pages.setdefault(b.page_index, []).append(b)

    for page_blocks in pages.values():
        for b in page_blocks:
            parent = _find_parent(b, page_blocks)
            if parent:
                b.parent_id = parent.block_id
                parent.child_ids.append(b.block_id)

        text_blocks = [x for x in page_blocks if x.block_type == BlockType.TEXT]
        formula_blocks = [x for x in page_blocks if x.block_type == BlockType.FORMULA]
        for f in formula_blocks:
            if f.parent_id:
                continue
            for t in text_blocks:
                if _contains(t, f, overlap_ratio=0.5):
                    f.parent_id = t.block_id
                    t.child_ids.append(f.block_id)
                    break

    return blocks

