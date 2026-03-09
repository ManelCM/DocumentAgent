from __future__ import annotations

from typing import Dict, List

from .types import DocumentBlock


def _column_key(block: DocumentBlock, median_width: float) -> int:
    # Heuristica estable para documentos en columnas: col izquierda/derecha.
    return 0 if block.bbox.cx < median_width else 1


def apply_reading_order(blocks: List[DocumentBlock]) -> List[DocumentBlock]:
    by_page: Dict[int, List[DocumentBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_index, []).append(b)

    for page_idx, page_blocks in by_page.items():
        if not page_blocks:
            continue

        max_x2 = max(b.bbox.x2 for b in page_blocks)
        median_width = max_x2 / 2.0
        ordered = sorted(page_blocks, key=lambda b: (_column_key(b, median_width), b.bbox.cy, b.bbox.cx))

        for rank, block in enumerate(ordered):
            block.reading_order = rank

    return blocks

