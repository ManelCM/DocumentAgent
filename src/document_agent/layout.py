from __future__ import annotations

from typing import List

from .types import BlockType, BoundingBox, DocumentBlock


def map_detector_label(label: str) -> BlockType:
    normalized = (label or "").strip().lower()
    if normalized in {"text", "title", "paragraph", "list", "reference"}:
        return BlockType.TEXT
    if normalized in {"formula", "equation", "inline_formula"}:
        return BlockType.FORMULA
    if normalized in {"table"}:
        return BlockType.OTHER
    if normalized in {"image", "photo"}:
        return BlockType.IMAGE
    if normalized in {"chart", "graph", "plot"}:
        return BlockType.CHART
    if normalized in {"figure"}:
        return BlockType.FIGURE
    if normalized in {"caption", "figure_caption"}:
        return BlockType.CAPTION
    return BlockType.OTHER


def detect_layout_blocks(pages: List, warnings: List[str]) -> List[DocumentBlock]:
    blocks: List[DocumentBlock] = []
    block_num = 0

    try:
        from paddleocr import LayoutDetection

        detector = LayoutDetection()
    except Exception as exc:  # pragma: no cover
        warnings.append(f"LayoutDetection unavailable. Using page-level fallback. Details: {exc}")
        detector = None

    for page_idx, img in enumerate(pages):
        page_h, page_w = img.shape[:2]
        if detector is None:
            block_num += 1
            blocks.append(
                DocumentBlock(
                    block_id=f"b{block_num}",
                    page_index=page_idx,
                    bbox=BoundingBox(0, 0, page_w, page_h),
                    block_type=BlockType.TEXT,
                    detector_label="page_fallback",
                    confidence=1.0,
                )
            )
            continue

        result = detector.predict(img)
        page_boxes = result[0].get("boxes", []) if result else []
        if not page_boxes:
            block_num += 1
            blocks.append(
                DocumentBlock(
                    block_id=f"b{block_num}",
                    page_index=page_idx,
                    bbox=BoundingBox(0, 0, page_w, page_h),
                    block_type=BlockType.TEXT,
                    detector_label="page_empty_fallback",
                    confidence=1.0,
                )
            )
            continue

        for item in page_boxes:
            x1, y1, x2, y2 = [int(v) for v in item["coordinate"]]
            block_num += 1
            label = str(item.get("label", "other"))
            blocks.append(
                DocumentBlock(
                    block_id=f"b{block_num}",
                    page_index=page_idx,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    block_type=map_detector_label(label),
                    detector_label=label,
                    confidence=float(item.get("score", 0.0)),
                )
            )

    return blocks

