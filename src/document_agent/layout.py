from __future__ import annotations

import threading
from typing import List

from .types import BlockType, BoundingBox, DocumentBlock


def map_detector_label(label: str) -> BlockType:
    normalized = (label or "").strip().lower()

    # ── Text family ───────────────────────────────────────────────────────────
    if normalized in {
        "text", "paragraph", "list", "reference", "abstract", "section",
        "reference_content",
        "doc_title",
        "aside_text",
    }:
        return BlockType.TEXT

    if normalized in {
        "title", "heading", "headline", "section_title", "sub_title",
        "paragraph_title",
    }:
        return BlockType.TEXT

    # ── Formula family ────────────────────────────────────────────────────────
    if normalized in {"formula", "equation", "inline_formula", "display_formula"}:
        return BlockType.FORMULA

    if normalized in {"formula_number"}:
        return BlockType.TEXT

    # ── Table ─────────────────────────────────────────────────────────────────
    if normalized in {"table", "table_caption"}:
        return BlockType.TABLE

    # ── Image / chart / figure ────────────────────────────────────────────────
    if normalized in {"image", "photo"}:
        return BlockType.IMAGE

    if normalized in {"chart", "graph", "plot"}:
        return BlockType.CHART

    if normalized in {"figure", "subfigure"}:
        return BlockType.FIGURE

    # ── Caption ───────────────────────────────────────────────────────────────
    if normalized in {
        "caption", "figure_caption", "table_caption_text",
        "figure_title",
    }:
        return BlockType.CAPTION

    # ── Header / Footer ───────────────────────────────────────────────────────
    if normalized in {"header", "page_header", "running_head", "running_title"}:
        return BlockType.HEADER

    if normalized in {
        "footer", "page_footer", "footnote", "page_number",
        "number",
    }:
        return BlockType.FOOTER

    return BlockType.OTHER


# ── Lazy singleton for LayoutDetection ───────────────────────────────────────
# Must be initialised in the main thread (before LangGraph starts its executor)
# to avoid a crash in PaddlePaddle's oneDNN/TBB backend when model loading
# happens concurrently with active threads.

_LAYOUT_DETECTOR = None
_LAYOUT_DETECTOR_LOADED = False
_LAYOUT_DETECTOR_LOCK = threading.Lock()


def get_layout_detector(warnings: List[str]):
    """Return the shared LayoutDetection instance, initialising it once."""
    global _LAYOUT_DETECTOR, _LAYOUT_DETECTOR_LOADED
    if _LAYOUT_DETECTOR_LOADED:
        return _LAYOUT_DETECTOR
    with _LAYOUT_DETECTOR_LOCK:
        if _LAYOUT_DETECTOR_LOADED:
            return _LAYOUT_DETECTOR
        _LAYOUT_DETECTOR_LOADED = True
        try:
            from paddleocr import LayoutDetection
            _LAYOUT_DETECTOR = LayoutDetection()
        except Exception as exc:
            warnings.append(
                f"LayoutDetection unavailable. Using page-level fallback. Details: {exc}"
            )
            _LAYOUT_DETECTOR = None
    return _LAYOUT_DETECTOR


def detect_layout_blocks(pages: List, warnings: List[str]) -> List[DocumentBlock]:
    """Detect layout blocks using PaddleOCR LayoutDetection (in-process).

    Both LayoutDetection and PaddleOCR OCR share PaddlePaddle's oneDNN/TBB
    backend.  They must be initialised in the main thread *before* LangGraph
    starts its executor threads (see cli.py pre-warm section).
    """
    blocks: List[DocumentBlock] = []
    block_num = 0

    detector = get_layout_detector(warnings)

    for page_idx, img in enumerate(pages):
        page_h, page_w = img.shape[:2]
        if detector is None:
            block_num += 1
            blocks.append(
                DocumentBlock(
                    block_id=f"p{page_idx}_b{block_num}",
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
