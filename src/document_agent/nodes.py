from __future__ import annotations

import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2

from .hierarchy import build_hierarchy
from .io import load_document_pages
from .layout import detect_layout_blocks
from .order import apply_reading_order
from .types import AgentState, BlockType, DocumentBlock
from .vlm_openai import (
    analyze_chart,
    analyze_formula,
    analyze_image,
    analyze_mixed,
    analyze_table,
    load_vlm_config,
)


# ──────────────────────────────────────────────────────────────────────────────
# Document loading
# ──────────────────────────────────────────────────────────────────────────────

def node_load_document(state: AgentState) -> AgentState:
    run_id = state.get("run_id") or str(uuid.uuid4())
    warnings = list(state.get("warnings", []))
    pages, sizes = load_document_pages(state["input_path"])
    return {
        **state,
        "run_id": run_id,
        "pages": pages,
        "page_sizes": sizes,
        "warnings": warnings,
        "status": "running",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Layout detection
# ──────────────────────────────────────────────────────────────────────────────

def node_detect_layout(state: AgentState) -> AgentState:
    warnings = list(state.get("warnings", []))
    blocks = detect_layout_blocks(state["pages"], warnings)
    return {**state, "blocks": blocks, "warnings": warnings}


# ──────────────────────────────────────────────────────────────────────────────
# Reading order
# ──────────────────────────────────────────────────────────────────────────────

def node_reading_order(state: AgentState) -> AgentState:
    blocks = apply_reading_order(state.get("blocks", []), state.get("page_sizes"))
    return {**state, "blocks": blocks}


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchy + inline math stitching
# ──────────────────────────────────────────────────────────────────────────────

def node_hierarchy(state: AgentState) -> AgentState:
    blocks = build_hierarchy(state.get("blocks", []))
    return {**state, "blocks": blocks}


# ──────────────────────────────────────────────────────────────────────────────
# Image cropping
# ──────────────────────────────────────────────────────────────────────────────

def _crop_block_image(state: AgentState, block_id: str):
    pages = state["pages"]
    block = next(b for b in state["blocks"] if b.block_id == block_id)
    img = pages[block.page_index]
    return img[block.bbox.y1: block.bbox.y2, block.bbox.x1: block.bbox.x2]


# ──────────────────────────────────────────────────────────────────────────────
# PaddleOCR text extraction (replaces Tesseract)
# ──────────────────────────────────────────────────────────────────────────────

_PADDLE_OCR = None
_PADDLE_OCR_LOADED = False


def _get_paddle_ocr():
    global _PADDLE_OCR, _PADDLE_OCR_LOADED
    if _PADDLE_OCR_LOADED:
        return _PADDLE_OCR
    _PADDLE_OCR_LOADED = True
    try:
        from paddleocr import PaddleOCR
        _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception:
        _PADDLE_OCR = None
    return _PADDLE_OCR


def _extract_text_paddle(crop) -> Dict:
    """Use PaddleOCR for text recognition. Falls back to Tesseract if unavailable."""
    if crop is None or not crop.size:
        return {"text": "", "ocr_engine": "paddleocr"}

    ocr = _get_paddle_ocr()
    if ocr is not None:
        try:
            result = ocr.predict(crop)
            lines = []
            if result:
                for page_result in result:
                    # New API returns objects/dicts with rec_texts
                    if hasattr(page_result, "rec_texts"):
                        lines.extend(page_result.rec_texts)
                    elif isinstance(page_result, list):
                        for line in page_result:
                            if line and len(line) >= 2 and line[1]:
                                lines.append(line[1][0])
            return {"text": " ".join(lines).strip(), "ocr_engine": "paddleocr"}
        except Exception:
            pass  # fall through to Tesseract

    # Tesseract fallback
    try:
        import pytesseract
        text = pytesseract.image_to_string(crop)
        return {"text": text.strip(), "ocr_engine": "pytesseract_fallback"}
    except Exception as exc:
        return {"text": "", "ocr_engine": "fallback", "ocr_error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# Block task dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _process_block_task(task: Dict) -> Dict:
    block: DocumentBlock = task["block"]
    crop = task["crop"]
    cfg = task["cfg"]
    kind: str = task["kind"]
    payload = {}
    warnings: List[str] = []

    if kind == "text":
        payload = _extract_text_paddle(crop)

    elif kind == "mixed":
        # Inline math + surrounding text — send merged crop to VLM
        if crop is None or not crop.size:
            payload = {"full_text": "", "formula_latex": "", "plain_text": "", "vlm_engine": "openai"}
        else:
            try:
                text, model_name = analyze_mixed(crop, cfg)
                payload = {"full_text": text, "vlm_engine": "openai", "vlm_model": model_name}
            except Exception as exc:
                # Fallback: OCR the merged crop
                ocr_result = _extract_text_paddle(crop)
                payload = {
                    "full_text": ocr_result.get("text", ""),
                    "vlm_engine": "fallback",
                    "vlm_model": "",
                }
                warnings.append(f"mixed_node fallback for {block.block_id}: {exc}")

    elif kind == "image":
        h, w = (crop.shape[:2] if crop is not None and crop.size else (0, 0))
        if crop is None or not crop.size:
            payload = {"description": "", "vlm_engine": "openai", "vlm_model": cfg.image_model}
        else:
            try:
                text, model_name = analyze_image(crop, cfg)
                payload = {"description": text, "vlm_engine": "openai", "vlm_model": model_name}
            except Exception as exc:
                payload = {
                    "description": f"Image region {w}x{h}. OpenAI VLM unavailable.",
                    "vlm_engine": "fallback",
                    "vlm_model": "",
                }
                warnings.append(f"image_node fallback for {block.block_id}: {exc}")

    elif kind == "chart":
        if crop is None or not crop.size:
            payload = {"chart_semantics": {"status": "empty_crop"}, "chart_engine": "openai"}
        else:
            try:
                text, model_name = analyze_chart(crop, cfg)
                payload = {
                    "chart_semantics": {"status": "ok", "data": text},
                    "chart_engine": "openai",
                    "chart_model": model_name,
                }
            except Exception as exc:
                payload = {"chart_semantics": {"status": "fallback"}, "chart_engine": "fallback"}
                warnings.append(f"chart_node fallback for {block.block_id}: {exc}")

    elif kind == "formula":
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop is not None and crop.size else crop
        try:
            text, model_name = analyze_formula(crop, cfg)
            payload = {
                "formula_latex": text,
                "formula_engine": "openai",
                "formula_model": model_name,
                "preprocess": {"shape": tuple(gray.shape) if gray is not None else (0, 0)},
            }
        except Exception as exc:
            payload = {
                "formula_latex": "",
                "formula_engine": "fallback",
                "formula_model": "",
                "preprocess": {"shape": tuple(gray.shape) if gray is not None else (0, 0)},
            }
            warnings.append(f"formula_node fallback for {block.block_id}: {exc}")

    elif kind == "other":
        # Unknown block type — run OCR so the payload is never empty.
        payload = _extract_text_paddle(crop)

    elif kind == "table":
        if crop is None or not crop.size:
            payload = {"table_data": {"status": "empty_crop"}, "table_engine": "openai"}
        else:
            try:
                text, model_name = analyze_table(crop, cfg)
                payload = {
                    "table_data": {"status": "ok", "data": text},
                    "table_engine": "openai",
                    "table_model": model_name,
                }
            except Exception as exc:
                # Fallback: OCR the table as plain text
                ocr_result = _extract_text_paddle(crop)
                payload = {
                    "table_data": {"status": "fallback", "data": ocr_result.get("text", "")},
                    "table_engine": "fallback",
                }
                warnings.append(f"table_node fallback for {block.block_id}: {exc}")

    return {"block_id": block.block_id, "payload": payload, "warnings": warnings}


# ──────────────────────────────────────────────────────────────────────────────
# Generic specialist runner
# ──────────────────────────────────────────────────────────────────────────────

_KIND_TO_TYPES: Dict[str, set] = {
    "text": {BlockType.TEXT},
    "mixed": {BlockType.MIXED},
    "image": {BlockType.IMAGE, BlockType.FIGURE},
    "chart": {BlockType.CHART},
    "formula": {BlockType.FORMULA},
    "table": {BlockType.TABLE},
    "other": {BlockType.CAPTION, BlockType.HEADER, BlockType.FOOTER, BlockType.OTHER},
}


def _run_specialist_kind(state: AgentState, kind: str) -> Tuple[Dict, List[str]]:
    blocks = state.get("blocks", [])
    cfg = load_vlm_config()
    target_types = _KIND_TO_TYPES.get(kind, set())
    tasks = []

    for block in blocks:
        if block.block_type not in target_types:
            continue
        if block.skip_specialist:
            continue
        crop = _crop_block_image(state, block.block_id)
        tasks.append({"kind": kind, "block": block, "crop": crop, "cfg": cfg})

    if not tasks:
        return {}, []

    max_workers = max(1, min(int(os.getenv("DOCAGENT_MAX_WORKERS", "4")), len(tasks)))
    updates: Dict[str, Dict] = {}
    warnings: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_block_task, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            updates[result["block_id"]] = result["payload"]
            warnings.extend(result["warnings"])

    return updates, warnings


# ──────────────────────────────────────────────────────────────────────────────
# Specialist nodes
# ──────────────────────────────────────────────────────────────────────────────

def node_text_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "text")
    return {"text_updates": updates, "text_warnings": warns}


def node_mixed_specialist(state: AgentState) -> AgentState:
    # MIXED blocks use their own keys to avoid collision with text_specialist
    # in the parallel fan-out. node_reduce_specialists merges them together.
    updates, warns = _run_specialist_kind(state, "mixed")
    return {"mixed_updates": updates, "mixed_warnings": warns}


def node_image_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "image")
    return {"image_updates": updates, "image_warnings": warns}


def node_chart_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "chart")
    return {"chart_updates": updates, "chart_warnings": warns}


def node_formula_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "formula")
    return {"formula_updates": updates, "formula_warnings": warns}


def node_table_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "table")
    return {"table_updates": updates, "table_warnings": warns}


def node_other_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "other")
    return {"other_updates": updates, "other_warnings": warns}


# ──────────────────────────────────────────────────────────────────────────────
# Reduce: merge all specialist payloads into blocks
# ──────────────────────────────────────────────────────────────────────────────

def node_reduce_specialists(state: AgentState) -> AgentState:
    blocks = state.get("blocks", [])
    by_id = {b.block_id: b for b in blocks}
    warnings = list(state.get("warnings", []))

    update_keys = [
        "text_updates", "mixed_updates", "image_updates", "chart_updates",
        "formula_updates", "table_updates", "other_updates",
    ]
    for key in update_keys:
        for block_id, payload in (state.get(key, {}) or {}).items():
            if block_id in by_id:
                by_id[block_id].payload.update(payload)

    warn_keys = [
        "text_warnings", "mixed_warnings", "image_warnings", "chart_warnings",
        "formula_warnings", "table_warnings", "other_warnings",
    ]
    for key in warn_keys:
        warnings.extend(state.get(key, []) or [])

    return {**state, "blocks": blocks, "warnings": warnings}


# ──────────────────────────────────────────────────────────────────────────────
# Cross-reference detection
# Scans OCR'd text blocks for "Figure X", "Table X", "Equation X", "[N]" etc.
# ──────────────────────────────────────────────────────────────────────────────

_XREF_PATTERNS = [
    (re.compile(r"\bfig(?:ure)?\.?\s*(\d+[a-z]?)\b", re.I), "figure"),
    (re.compile(r"\btable\.?\s*(\d+[a-z]?)\b", re.I), "table"),
    (re.compile(r"\beq(?:uation)?\.?\s*(\d+[a-z]?)\b", re.I), "equation"),
    (re.compile(r"\bsec(?:tion)?\.?\s*(\d+[\.\d]*)\b", re.I), "section"),
    (re.compile(r"\[(\d+)\]", re.I), "citation"),
]


def node_cross_reference(state: AgentState) -> AgentState:
    """
    Scan text block payloads for references to figures, tables, equations, and citations.
    Store found references in block.relations["references"].
    """
    blocks = state.get("blocks", [])

    for block in blocks:
        if block.block_type not in {BlockType.TEXT, BlockType.MIXED}:
            continue
        text = block.payload.get("text") or block.payload.get("full_text") or ""
        if not text:
            continue

        refs = []
        for pattern, ref_type in _XREF_PATTERNS:
            for match in pattern.finditer(text):
                refs.append({"type": ref_type, "label": match.group(1), "context": match.group(0)})

        if refs:
            block.relations["references"] = refs

    return {**state, "blocks": blocks}


# ──────────────────────────────────────────────────────────────────────────────
# Association: captions ↔ figures, full-page images, embedded formulas
# ──────────────────────────────────────────────────────────────────────────────

def node_association(state: AgentState) -> AgentState:
    blocks = state.get("blocks", [])
    captions = [b for b in blocks if b.block_type == BlockType.CAPTION]
    figure_like = [b for b in blocks if b.block_type in {BlockType.FIGURE, BlockType.IMAGE, BlockType.CHART, BlockType.TABLE}]

    for cap in captions:
        same_page = [f for f in figure_like if f.page_index == cap.page_index]
        if not same_page:
            continue
        # Skip if already linked by hierarchy
        if cap.relations.get("describes"):
            continue
        nearest = min(
            same_page,
            key=lambda fig: abs(fig.bbox.cy - cap.bbox.cy) + abs(fig.bbox.cx - cap.bbox.cx),
        )
        cap.relations["describes"] = nearest.block_id
        nearest.relations.setdefault("captions", []).append(cap.block_id)

    # Full-page image/figure detection
    page_sizes = {p["page_index"]: p for p in state.get("page_sizes", [])}
    for fig in figure_like:
        page = page_sizes.get(fig.page_index)
        if not page:
            continue
        page_area = max(1, int(page["width"]) * int(page["height"]))
        coverage = fig.bbox.area / page_area
        if coverage >= 0.85:
            fig.relations["is_page_image"] = True
            fig.relations["page_coverage"] = round(float(coverage), 4)

    # Embedded formulas (those absorbed into MIXED blocks)
    id_to_block = {b.block_id: b for b in blocks}
    formulas = [b for b in blocks if b.block_type == BlockType.FORMULA]
    for formula in formulas:
        if formula.parent_id and formula.parent_id in id_to_block:
            parent = id_to_block[formula.parent_id]
            if parent.block_type in {BlockType.TEXT, BlockType.MIXED}:
                parent.relations.setdefault("embedded_formulas", []).append(formula.block_id)
                formula.relations["embedded_in"] = parent.block_id

    return {**state, "blocks": blocks}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate: build final structured output
# ──────────────────────────────────────────────────────────────────────────────

def node_aggregate(state: AgentState) -> AgentState:
    blocks = state.get("blocks", [])
    ordered = sorted(
        blocks,
        key=lambda b: (b.page_index, b.reading_order if b.reading_order >= 0 else 999_999),
    )
    type_counts: Dict[str, int] = {}
    for b in ordered:
        type_counts[b.block_type.value] = type_counts.get(b.block_type.value, 0) + 1

    output: Dict = {
        "run_id": state.get("run_id"),
        "input_path": state.get("input_path"),
        "status": "done",
        "warnings": state.get("warnings", []),
        "pages": state.get("page_sizes", []),
        "blocks": [b.as_dict() for b in ordered],
        "summary": {
            "num_pages": len(state.get("page_sizes", [])),
            "num_blocks": len(ordered),
            "types": type_counts,
        },
    }
    return {**state, "output": output, "status": "done"}
