from __future__ import annotations

import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2

from .hierarchy import build_hierarchy
from .io import load_document_pages
from .layout import detect_layout_blocks
from .order import apply_reading_order
from .tracer import finish_trace, init_trace, node_end, node_start, record_block, record_llm_call
from .vlm_openai import PROMPT_REGISTRY
from .types import AgentState, BlockType, DocumentBlock
from .vlm_openai import (
    analyze_chart,
    analyze_formula,
    analyze_image,
    analyze_mixed,
    analyze_table,
    load_vlm_config,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Document loading
# ──────────────────────────────────────────────────────────────────────────────

def node_load_document(state: AgentState) -> AgentState:
    trace = init_trace()
    entry = node_start(trace, "load_document")
    run_id = state.get("run_id") or str(uuid.uuid4())
    warnings = list(state.get("warnings", []))
    logger.info("[load_document] loading %s", state["input_path"])
    pages, sizes = load_document_pages(state["input_path"])

    node_end(entry)
    logger.info("[load_document] %d pages loaded in %.2fs", len(pages), entry["duration_s"])
    return {
        **state,
        "run_id": run_id,
        "pages": pages,
        "page_sizes": sizes,
        "warnings": warnings,
        "status": "running",
        "_trace": trace,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Layout detection
# ──────────────────────────────────────────────────────────────────────────────

def _iou(a: "DocumentBlock", b: "DocumentBlock") -> float:
    """Intersection-over-Union of two bounding boxes."""
    ix1 = max(a.bbox.x1, b.bbox.x1)
    iy1 = max(a.bbox.y1, b.bbox.y1)
    ix2 = min(a.bbox.x2, b.bbox.x2)
    iy2 = min(a.bbox.y2, b.bbox.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = a.bbox.area + b.bbox.area - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_blocks(
    blocks: List[DocumentBlock],
    iou_threshold: float = 0.5,
) -> List[DocumentBlock]:
    """Remove duplicate / heavily-overlapping blocks on the same page.

    When two blocks of the *same* type overlap with IoU ≥ iou_threshold,
    keep whichever has the higher detector confidence (the other is dropped).
    Cross-type overlaps are left to the hierarchy node to resolve via
    containment/absorption logic.

    Typical source of duplicates: some layout models fire two bounding boxes
    for the same region at slightly different scales (common on scanned docs).
    """
    if not blocks:
        return blocks

    # Group by page to reduce O(n²) comparisons
    by_page: Dict[int, List[DocumentBlock]] = {}
    for b in blocks:
        by_page.setdefault(b.page_index, []).append(b)

    kept: List[DocumentBlock] = []
    for page_blocks in by_page.values():
        # Sort descending by confidence so higher-confidence blocks are kept first
        sorted_page = sorted(page_blocks, key=lambda b: b.confidence, reverse=True)
        accepted: List[DocumentBlock] = []
        for candidate in sorted_page:
            duplicate = False
            for existing in accepted:
                if (existing.block_type == candidate.block_type
                        and _iou(candidate, existing) >= iou_threshold):
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(candidate)
        kept.extend(accepted)

    return kept


def node_detect_layout(state: AgentState) -> AgentState:
    trace = state.get("_trace", init_trace())
    entry = node_start(trace, "detect_layout")
    warnings = list(state.get("warnings", []))
    blocks = detect_layout_blocks(state["pages"], warnings)

    # Confidence threshold: drop detector results below min confidence.
    # Fallback blocks (confidence=1.0, label="page_fallback") are never dropped.
    # Default 0.3 catches garbage detections on noisy/scanned documents without
    # discarding real content. Override with DOCAGENT_MIN_CONFIDENCE env var.
    import os as _os
    min_conf = float(_os.getenv("DOCAGENT_MIN_CONFIDENCE", "0.3"))
    kept, dropped = [], []
    for b in blocks:
        if b.detector_label in {"page_fallback", "page_empty_fallback"} or b.confidence >= min_conf:
            kept.append(b)
        else:
            dropped.append(b)
    if dropped:
        warnings.append(
            f"[detect_layout] dropped {len(dropped)} low-confidence blocks "
            f"(threshold={min_conf})"
        )
    blocks = kept

    # Deduplication: remove same-type overlapping blocks (scanned-doc artefacts).
    # Override with DOCAGENT_DEDUP_IOU=0 to disable.
    iou_threshold = float(_os.getenv("DOCAGENT_DEDUP_IOU", "0.5"))
    if iou_threshold > 0:
        before = len(blocks)
        blocks = _deduplicate_blocks(blocks, iou_threshold)
        n_removed = before - len(blocks)
        if n_removed:
            warnings.append(
                f"[detect_layout] removed {n_removed} duplicate blocks "
                f"(IoU≥{iou_threshold})"
            )

    node_end(entry)
    logger.info("[detect_layout] %d blocks detected in %.2fs", len(blocks), entry["duration_s"])
    return {**state, "blocks": blocks, "warnings": warnings, "_trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# Reading order
# ──────────────────────────────────────────────────────────────────────────────

def node_reading_order(state: AgentState) -> AgentState:
    trace = state.get("_trace", init_trace())
    entry = node_start(trace, "reading_order")
    blocks = apply_reading_order(state.get("blocks", []), state.get("page_sizes"))
    node_end(entry)
    logger.info("[reading_order] order assigned in %.2fs", entry["duration_s"])
    return {**state, "blocks": blocks, "_trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchy + inline math stitching
# ──────────────────────────────────────────────────────────────────────────────

def node_hierarchy(state: AgentState) -> AgentState:
    trace = state.get("_trace", init_trace())
    entry = node_start(trace, "hierarchy")
    blocks = build_hierarchy(state.get("blocks", []))
    node_end(entry)
    mixed = sum(1 for b in blocks if b.block_type == BlockType.MIXED)
    logger.info("[hierarchy] done in %.2fs - %d MIXED blocks stitched", entry["duration_s"], mixed)
    return {**state, "blocks": blocks, "_trace": trace}


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

import threading as _threading

_PADDLE_OCR = None
_PADDLE_OCR_LOADED = False
_PADDLE_OCR_LOCK = _threading.Lock()


def _get_paddle_ocr():
    """PaddleOCR v3 PP-OCRv5 server models conflict with LayoutDetection in-process
    on Windows (PaddlePaddle oneDNN/TBB backend crashes with SIGSEGV when both sets
    of models are loaded in the same process).  Text blocks use pytesseract instead,
    which gives excellent quality for clean academic/document text.
    """
    global _PADDLE_OCR, _PADDLE_OCR_LOADED
    if _PADDLE_OCR_LOADED:
        return _PADDLE_OCR
    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_LOADED:
            return _PADDLE_OCR
        _PADDLE_OCR_LOADED = True
        _PADDLE_OCR = None  # Always use pytesseract fallback
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
                    # PaddleOCR v3 returns a list of dicts with 'rec_texts' key
                    if isinstance(page_result, dict) and "rec_texts" in page_result:
                        lines.extend(page_result["rec_texts"])
                    elif hasattr(page_result, "rec_texts"):
                        lines.extend(page_result.rec_texts)
                    elif isinstance(page_result, list):
                        # Old PaddleOCR v2 format: [[bbox, (text, score)], ...]
                        for line in page_result:
                            if line and len(line) >= 2 and line[1]:
                                lines.append(line[1][0])
            return {"text": " ".join(lines).strip(), "ocr_engine": "paddleocr"}
        except Exception:
            pass  # fall through to Tesseract

    # Tesseract (primary text engine — PaddleOCR intentionally disabled)
    try:
        import pytesseract
        # Use image_to_data to get per-word confidence scores.
        # Words with confidence < threshold are dropped (catches OCR noise).
        min_conf = int(os.getenv("DOCAGENT_TESSERACT_MIN_CONF", "30"))
        data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT)
        words, confs = data["text"], data["conf"]
        kept = [w for w, c in zip(words, confs) if str(c).lstrip("-").isdigit() and int(c) >= min_conf and w.strip()]
        text = " ".join(kept)
        # Compute average confidence over all detected words (not just kept ones)
        valid_confs = [int(c) for c in confs if str(c).lstrip("-").isdigit() and int(c) >= 0]
        avg_conf = round(sum(valid_confs) / len(valid_confs) / 100.0, 3) if valid_confs else 0.0
        return {"text": text.strip(), "ocr_engine": "pytesseract", "ocr_confidence": avg_conf}
    except Exception as exc:
        return {"text": "", "ocr_engine": "fallback", "ocr_error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# Block task dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _process_block_task(task: Dict) -> Dict:
    """Process one block.  Returns payload + full LLM audit trail."""
    t0 = time.time()
    block: DocumentBlock = task["block"]
    crop   = task["crop"]
    cfg    = task["cfg"]
    kind: str = task["kind"]
    payload: Dict = {}
    warnings: List[str] = []
    # Buffered LLM call records — written to trace in main thread after return
    llm_calls: List[Dict] = []

    import json as _json

    def _parse_vlm(raw: str, defaults: Dict) -> Dict:
        """Parse the VLM JSON response into a flat dict; fall back to defaults."""
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return defaults

    if kind == "text":
        payload = _extract_text_paddle(crop)

    elif kind == "mixed":
        if crop is None or not crop.size:
            payload = {"full_text": "", "formula_latex": "", "plain_text": "", "vlm_engine": "openai"}
        else:
            try:
                t_call = time.time()
                text, model_name, usage, prompt_name = analyze_mixed(crop, cfg)
                call_dur = time.time() - t_call
                parsed = _parse_vlm(text, {"full_text": text, "formula_latex": "", "plain_text": ""})
                payload = {
                    "full_text":     parsed.get("full_text", text),
                    "formula_latex": parsed.get("formula_latex", ""),
                    "plain_text":    parsed.get("plain_text", ""),
                    "confidence":    parsed.get("confidence", 0.0),
                    "vlm_engine":    "openai",
                    "vlm_model":     model_name,
                    "_raw_response": text,
                }
                llm_calls.append({"model": model_name, "prompt_name": prompt_name,
                                   "response_text": text, "usage": usage, "duration_s": call_dur})
            except Exception as exc:
                ocr_result = _extract_text_paddle(crop)
                payload = {"full_text": ocr_result.get("text", ""), "vlm_engine": "fallback", "vlm_model": ""}
                warnings.append(f"mixed fallback for {block.block_id}: {exc}")

    elif kind == "image":
        h, w = (crop.shape[:2] if crop is not None and crop.size else (0, 0))
        if crop is None or not crop.size:
            payload = {"summary": "", "vlm_engine": "openai", "vlm_model": cfg.image_model}
        else:
            try:
                t_call = time.time()
                text, model_name, usage, prompt_name = analyze_image(crop, cfg)
                call_dur = time.time() - t_call
                parsed = _parse_vlm(text, {"summary": text})
                payload = {
                    "summary":       parsed.get("summary", text),
                    "key_elements":  parsed.get("key_elements", []),
                    "visible_text":  parsed.get("visible_text", ""),
                    "spatial_layout": parsed.get("spatial_layout", ""),
                    "confidence":    parsed.get("confidence", 0.0),
                    "vlm_engine":    "openai",
                    "vlm_model":     model_name,
                    "_raw_response": text,
                }
                llm_calls.append({"model": model_name, "prompt_name": prompt_name,
                                   "response_text": text, "usage": usage, "duration_s": call_dur})
            except Exception as exc:
                payload = {"summary": f"Image region {w}×{h}. VLM unavailable.", "vlm_engine": "fallback", "vlm_model": ""}
                warnings.append(f"image fallback for {block.block_id}: {exc}")

    elif kind == "chart":
        if crop is None or not crop.size:
            payload = {"takeaway": "", "chart_engine": "openai"}
        else:
            try:
                t_call = time.time()
                text, model_name, usage, prompt_name = analyze_chart(crop, cfg)
                call_dur = time.time() - t_call
                parsed = _parse_vlm(text, {"takeaway": text})
                payload = {
                    "chart_type": parsed.get("chart_type", ""),
                    "title":      parsed.get("title", ""),
                    "axes":       parsed.get("axes", {}),
                    "series":     parsed.get("series", []),
                    "trends":     parsed.get("trends", ""),
                    "extrema":    parsed.get("extrema", {}),
                    "comparisons": parsed.get("comparisons", ""),
                    "data_table": parsed.get("data_table", ""),
                    "takeaway":   parsed.get("takeaway", ""),
                    "confidence": parsed.get("confidence", 0.0),
                    "chart_engine": "openai",
                    "chart_model":  model_name,
                    "_raw_response": text,
                }
                llm_calls.append({"model": model_name, "prompt_name": prompt_name,
                                   "response_text": text, "usage": usage, "duration_s": call_dur})
            except Exception as exc:
                payload = {"takeaway": "", "chart_engine": "fallback"}
                warnings.append(f"chart fallback for {block.block_id}: {exc}")

    elif kind == "formula":
        if crop is None or not crop.size:
            payload = {"latex": "", "formula_engine": "openai"}
        else:
            try:
                t_call = time.time()
                text, model_name, usage, prompt_name = analyze_formula(crop, cfg)
                call_dur = time.time() - t_call
                parsed = _parse_vlm(text, {"latex": text})
                payload = {
                    "latex":        parsed.get("latex", text),
                    "symbols":      parsed.get("symbols", []),
                    "formula_type": parsed.get("formula_type", ""),
                    "meaning":      parsed.get("meaning", ""),
                    "confidence":   parsed.get("confidence", 0.0),
                    "formula_engine": "openai",
                    "formula_model":  model_name,
                    "_raw_response":  text,
                }
                llm_calls.append({"model": model_name, "prompt_name": prompt_name,
                                   "response_text": text, "usage": usage, "duration_s": call_dur})
            except Exception as exc:
                payload = {"latex": "", "formula_engine": "fallback", "formula_model": ""}
                warnings.append(f"formula fallback for {block.block_id}: {exc}")

    elif kind == "other":
        payload = _extract_text_paddle(crop)

    elif kind == "table":
        if crop is None or not crop.size:
            payload = {"summary": "", "headers": [], "rows": [], "table_engine": "openai"}
        else:
            try:
                t_call = time.time()
                text, model_name, usage, prompt_name = analyze_table(crop, cfg)
                call_dur = time.time() - t_call
                parsed = _parse_vlm(text, {"summary": text, "headers": [], "rows": []})
                payload = {
                    "title":        parsed.get("title", ""),
                    "headers":      parsed.get("headers", []),
                    "rows":         parsed.get("rows", []),
                    "notes":        parsed.get("notes", ""),
                    "summary":      parsed.get("summary", ""),
                    "confidence":   parsed.get("confidence", 0.0),
                    "table_engine": "openai",
                    "table_model":  model_name,
                    "_raw_response": text,
                }
                llm_calls.append({"model": model_name, "prompt_name": prompt_name,
                                   "response_text": text, "usage": usage, "duration_s": call_dur})
            except Exception as exc:
                ocr_result = _extract_text_paddle(crop)
                payload = {"summary": ocr_result.get("text", ""), "headers": [], "rows": [], "table_engine": "fallback"}
                warnings.append(f"table fallback for {block.block_id}: {exc}")

    duration = time.time() - t0
    engine = (
        payload.get("ocr_engine")
        or payload.get("vlm_engine")
        or payload.get("formula_engine")
        or payload.get("chart_engine")
        or payload.get("table_engine")
        or "unknown"
    )
    logger.debug("[%s] %s %s engine=%s t=%.3fs", kind, block.block_id, block.block_type.value, engine, duration)
    return {
        "block_id":  block.block_id,
        "page_index": block.page_index,
        "block_type": block.block_type.value,
        "payload":   payload,
        "warnings":  warnings,
        "llm_calls": llm_calls,   # buffered; recorded in main thread by _run_specialist_kind
        "_duration": duration,
        "_engine":   engine,
        "_kind":     kind,
    }


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
    trace = state.get("_trace") or {}
    cfg = load_vlm_config()
    target_types = _KIND_TO_TYPES.get(kind, set())
    block_map = {b.block_id: b for b in blocks}
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

    # Per-kind worker override, e.g. DOCAGENT_FORMULA_MAX_WORKERS=16
    kind_env = f"DOCAGENT_{kind.upper()}_MAX_WORKERS"
    global_default = int(os.getenv("DOCAGENT_MAX_WORKERS", "8"))
    desired = int(os.getenv(kind_env, str(global_default)))
    max_workers = max(1, min(desired, len(tasks)))
    updates: Dict[str, Dict] = {}
    warnings: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_block_task, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            bid = result["block_id"]
            updates[bid] = result["payload"]
            warnings.extend(result["warnings"])

            # ── Record block provenance in trace ─────────────────────────────
            if bid in block_map:
                record_block(
                    trace,
                    block_map[bid],
                    specialist=result.get("_kind", kind),
                    engine=result.get("_engine", "unknown"),
                    duration_s=result.get("_duration", 0.0),
                    payload=result["payload"],
                )

            # ── Record every LLM call (thread-safe via list.append / GIL) ───
            for call in result.get("llm_calls", []):
                record_llm_call(
                    trace,
                    block_id=bid,
                    page_index=result.get("page_index", -1),
                    block_type=result.get("block_type", kind),
                    specialist=result.get("_kind", kind),
                    model=call["model"],
                    prompt_name=call["prompt_name"],
                    prompt_text=PROMPT_REGISTRY.get(call["prompt_name"], call["prompt_name"]),
                    response_text=call["response_text"],
                    usage=call["usage"],
                    duration_s=call["duration_s"],
                )

    return updates, warnings


# ──────────────────────────────────────────────────────────────────────────────
# Specialist nodes
# ──────────────────────────────────────────────────────────────────────────────

def node_text_specialist(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "text_specialist")
    updates, warns = _run_specialist_kind(state, "text")
    node_end(entry)
    logger.info("[text_specialist] %d blocks in %.2fs", len(updates), entry["duration_s"])
    return {"text_updates": updates, "text_warnings": warns}


def node_mixed_specialist(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "mixed_specialist")
    updates, warns = _run_specialist_kind(state, "mixed")
    node_end(entry)
    logger.info("[mixed_specialist] %d blocks in %.2fs", len(updates), entry["duration_s"])
    return {"mixed_updates": updates, "mixed_warnings": warns}


def node_image_specialist(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "image_specialist")
    updates, warns = _run_specialist_kind(state, "image")
    node_end(entry)
    logger.info("[image_specialist] %d blocks in %.2fs", len(updates), entry["duration_s"])
    return {"image_updates": updates, "image_warnings": warns}


def node_chart_specialist(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "chart_specialist")
    updates, warns = _run_specialist_kind(state, "chart")
    node_end(entry)
    logger.info("[chart_specialist] %d blocks in %.2fs", len(updates), entry["duration_s"])
    return {"chart_updates": updates, "chart_warnings": warns}


def node_formula_specialist(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "formula_specialist")
    updates, warns = _run_specialist_kind(state, "formula")
    node_end(entry)
    logger.info("[formula_specialist] %d blocks in %.2fs", len(updates), entry["duration_s"])
    return {"formula_updates": updates, "formula_warnings": warns}


def node_table_specialist(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "table_specialist")
    updates, warns = _run_specialist_kind(state, "table")
    node_end(entry)
    logger.info("[table_specialist] %d blocks in %.2fs", len(updates), entry["duration_s"])
    return {"table_updates": updates, "table_warnings": warns}


def node_other_specialist(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "other_specialist")
    updates, warns = _run_specialist_kind(state, "other")
    node_end(entry)
    logger.info("[other_specialist] %d blocks in %.2fs", len(updates), entry["duration_s"])
    return {"other_updates": updates, "other_warnings": warns}


# ──────────────────────────────────────────────────────────────────────────────
# Reduce: merge all specialist payloads into blocks
# ──────────────────────────────────────────────────────────────────────────────

def node_reduce_specialists(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "reduce_specialists")
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

    node_end(entry)
    logger.info("[reduce_specialists] merged in %.2fs - %d warnings total",
                entry["duration_s"], len(warnings))
    return {**state, "blocks": blocks, "warnings": warnings, "_trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# Cross-reference detection + resolution
# Scans OCR'd text blocks for "Figure X", "Table X", "Equation X", "[N]" etc.
# Then resolves each mention to the actual block it refers to.
# ──────────────────────────────────────────────────────────────────────────────

_XREF_PATTERNS = [
    (re.compile(r"\bfig(?:ure)?\.?\s*(\d+[a-z]?)\b", re.I), "figure"),
    (re.compile(r"\btable\.?\s*(\d+[a-z]?)\b", re.I), "table"),
    (re.compile(r"\beq(?:uation)?\.?\s*(\d+[a-z]?)\b", re.I), "equation"),
    (re.compile(r"\bsec(?:tion)?\.?\s*(\d+[\.\d]*)\b", re.I), "section"),
    (re.compile(r"\balg(?:orithm)?\.?\s*(\d+[a-z]?)\b", re.I), "algorithm"),
    (re.compile(r"\blisting\.?\s*(\d+[a-z]?)\b", re.I), "listing"),
    (re.compile(r"\[(\d+)\]", re.I), "citation"),
]

# Caption-start patterns used to build the label→block index
_CAPTION_LABEL_PATTERN = re.compile(
    r"^(fig(?:ure)?|table|eq(?:uation)?|algorithm|alg|listing)\s*\.?\s*(\d+[a-z]?)\b",
    re.I,
)
_CAPTION_TYPE_MAP = {
    "fig": "figure", "figure": "figure",
    "table": "table",
    "eq": "equation", "equation": "equation",
    "algorithm": "algorithm", "alg": "algorithm",
    "listing": "listing",
}


def _build_label_index(blocks: List[DocumentBlock]) -> Dict[Tuple[str, str], str]:
    """Return {(ref_type, label) → block_id} built from caption text.

    Looks for captions that start with "Figure 3", "Table 2", etc.
    Maps to the figure/table block the caption describes.
    """
    index: Dict[Tuple[str, str], str] = {}
    for block in blocks:
        if block.block_type != BlockType.CAPTION:
            continue
        text = (block.payload.get("text") or "").strip()
        m = _CAPTION_LABEL_PATTERN.match(text)
        if not m:
            continue
        raw_type = m.group(1).lower()
        ref_type = _CAPTION_TYPE_MAP.get(raw_type, raw_type)
        label = m.group(2).lower()
        # The caption's `describes` relation points to the figure/table block
        target_id = block.relations.get("describes")
        if target_id:
            index[(ref_type, label)] = target_id
        # Also index by the caption block itself (some refs point to captions)
        index[(ref_type, label + "_cap")] = block.block_id
    return index


def node_cross_reference(state: AgentState) -> AgentState:
    """Scan text payloads for references to figures, tables, equations, citations.

    Each found reference is resolved to a target block_id when possible, adding
    a ``target_id`` field to the reference dict.  The target block gains a
    ``referenced_by`` list so the graph is navigable in both directions.
    """
    trace = state.get("_trace") or {}
    entry = node_start(trace, "cross_reference")
    blocks = state.get("blocks", [])

    # Build label index from captions (requires association to have run first)
    label_index = _build_label_index(blocks)
    id_to_block = {b.block_id: b for b in blocks}

    total_refs = 0
    resolved = 0

    for block in blocks:
        if block.block_type not in {BlockType.TEXT, BlockType.MIXED}:
            continue
        text = block.payload.get("text") or block.payload.get("full_text") or ""
        if not text:
            continue

        refs = []
        for pattern, ref_type in _XREF_PATTERNS:
            for match in pattern.finditer(text):
                label = match.group(1).lower()
                ref = {"type": ref_type, "label": label, "context": match.group(0)}
                # Resolve to target block
                target_id = label_index.get((ref_type, label))
                if target_id:
                    ref["target_id"] = target_id
                    # Back-link on the target block
                    target = id_to_block.get(target_id)
                    if target:
                        target.relations.setdefault("referenced_by", [])
                        if block.block_id not in target.relations["referenced_by"]:
                            target.relations["referenced_by"].append(block.block_id)
                    resolved += 1
                refs.append(ref)

        if refs:
            block.relations["references"] = refs
            total_refs += len(refs)

    node_end(entry)
    logger.info(
        "[cross_reference] %d references found (%d resolved) in %.2fs",
        total_refs, resolved, entry["duration_s"],
    )
    return {**state, "blocks": blocks, "_trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# Association: captions ↔ figures, full-page images, embedded formulas
# ──────────────────────────────────────────────────────────────────────────────

def node_association(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "association")
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

    # Embedded formulas: map formula children onto any parent that can render them
    id_to_block = {b.block_id: b for b in blocks}
    formulas = [b for b in blocks if b.block_type == BlockType.FORMULA]
    for formula in formulas:
        if formula.parent_id and formula.parent_id in id_to_block:
            parent = id_to_block[formula.parent_id]
            if parent.block_type in {BlockType.TEXT, BlockType.MIXED,
                                     BlockType.TABLE, BlockType.CAPTION}:
                parent.relations.setdefault("embedded_formulas", []).append(formula.block_id)
                formula.relations["embedded_in"] = parent.block_id

    node_end(entry)
    logger.info("[association] linked %d captions in %.2fs", len(captions), entry["duration_s"])
    return {**state, "blocks": blocks, "_trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate: build final structured output
# ──────────────────────────────────────────────────────────────────────────────

import re as _re


def _dehyphenate(text: str) -> str:
    """Join words split across line-breaks by a hyphen.

    Handles both intra-column breaks ("dis-\ntillation") and
    inter-column breaks where a word ends the left column and
    continues at the start of the right column.
    """
    # Pattern: word-char, hyphen, newline (optional spaces), word-char
    return _re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)


def _build_full_text(blocks_ordered: List[DocumentBlock]) -> str:
    """Concatenate all content blocks in reading order for RAG / downstream use.

    Payload fields are now flat (parsed at ingest time), so no JSON re-parsing.
    Blocks that are children absorbed into parents are skipped (except their
    embedded formulas, which are appended after the parent text).
    Headers / footers / other boilerplate are skipped.
    Page breaks inserted as [PAGE N] markers.
    """
    # Build full block map so we can look up children (formula embeddings)
    block_map = {b.block_id: b for b in blocks_ordered}

    lines: List[str] = []
    current_page = -1

    for block in blocks_ordered:
        # Skip children absorbed into MIXED groups (parent renders them)
        if block.skip_specialist and block.parent_id:
            continue
        # Skip containment-only children (formula inside text block, etc.)
        if block.parent_id and block.block_type != BlockType.MIXED:
            continue
        # Skip boilerplate and vertical margin annotations
        if block.block_type in {BlockType.HEADER, BlockType.FOOTER, BlockType.OTHER}:
            continue
        if block.detector_label == "aside_text":
            continue
        # Skip noise: text blocks with no meaningful content (tiny margin artifacts)
        if block.block_type == BlockType.TEXT:
            raw_text = block.payload.get("text", "") or ""
            if len(raw_text.strip()) < 3:
                continue

        if block.page_index != current_page:
            current_page = block.page_index
            if lines:
                lines.append("")
            lines.append(f"[PAGE {current_page + 1}]")
            lines.append("")

        p = block.payload
        btype = block.block_type

        if btype == BlockType.TEXT:
            text = _dehyphenate(p.get("text", "") or "").strip()
            if text:
                lines.append(text)
            # Embed child formulas (already have LaTeX from formula specialist)
            for fid in sorted(
                block.relations.get("embedded_formulas", []),
                key=lambda fid: (
                    block_map[fid].bbox.y1 if fid in block_map else 0,
                    block_map[fid].bbox.x1 if fid in block_map else 0,
                ),
            ):
                child = block_map.get(fid)
                if not child:
                    continue
                fp = child.payload or {}
                latex = (fp.get("latex") or "").strip()
                meaning = (fp.get("meaning") or "").strip()
                if latex:
                    lines.append(f"  {{formula: {latex}}}")
                    if meaning:
                        lines.append(f"  [meaning: {meaning}]")

        elif btype == BlockType.MIXED:
            text = _dehyphenate(p.get("full_text", "") or "").strip()
            if text:
                lines.append(text)

        elif btype == BlockType.FORMULA:
            # VLM may return null for any field; use `or ""` before .strip()
            latex   = (p.get("latex")   or "").strip()
            meaning = (p.get("meaning") or "").strip()
            ftype   = (p.get("formula_type") or "").strip()
            if latex:
                lines.append(f"$$\n{latex}\n$$")
                if meaning:
                    lines.append(f"  [{ftype}: {meaning}]" if ftype else f"  [{meaning}]")
            else:
                lines.append("[FORMULA]")

        elif btype == BlockType.TABLE:
            def _s(v) -> str:
                return (v or "").strip() if not isinstance(v, (list, dict)) else ""
            summary     = _s(p.get("summary"))
            title       = _s(p.get("title"))
            headers     = p.get("headers") or []
            rows        = p.get("rows") or []
            row_headers = p.get("row_headers") or []
            notes       = _s(p.get("notes"))
            lines.append(f"[TABLE: {title or summary}]" if (title or summary) else "[TABLE]")
            if row_headers:
                lines.append(f"  Row labels: {', '.join(str(r) for r in row_headers)}")
            if headers and rows:
                lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                lines.append("|" + "|".join(" --- " for _ in headers) + "|")
                for row in rows:
                    cells = [str(row.get(h) if row.get(h) is not None else "") for h in headers]
                    lines.append("| " + " | ".join(cells) + " |")
            if notes:
                lines.append(f"_{notes}_")
            # Emit formulas that live inside this table (e.g. cell equations)
            for fid in sorted(
                block.relations.get("embedded_formulas", []),
                key=lambda fid: (
                    block_map[fid].bbox.y1 if fid in block_map else 0,
                    block_map[fid].bbox.x1 if fid in block_map else 0,
                ),
            ):
                child = block_map.get(fid)
                if not child:
                    continue
                fp = child.payload or {}
                latex = (fp.get("latex") or "").strip()
                meaning = (fp.get("meaning") or "").strip()
                if latex:
                    lines.append(f"  {{formula: {latex}}}")
                    if meaning:
                        lines.append(f"  [meaning: {meaning}]")

        elif btype in {BlockType.IMAGE, BlockType.FIGURE}:
            summary      = ((p.get("summary") or "").strip())
            visible_text = ((p.get("visible_text") or "").strip())
            key_elements = p.get("key_elements") or []
            lines.append(f"[IMAGE: {summary}]" if summary else "[IMAGE]")
            if visible_text:
                lines.append(f"  Visible text: {visible_text}")
            if key_elements:
                lines.append("  Key elements: " + ", ".join(str(e) for e in key_elements[:8]))

        elif btype == BlockType.CHART:
            def _sv(v) -> str:
                if isinstance(v, list):
                    return "; ".join(str(i) for i in v if i)
                return str(v).strip() if v else ""
            takeaway   = _sv(p.get("takeaway"))
            trends     = _sv(p.get("trends"))
            chart_type = _sv(p.get("chart_type"))
            data_table = _sv(p.get("data_table"))
            title      = _sv(p.get("title"))
            header = f"[CHART ({chart_type}): {title}]" if title else f"[CHART ({chart_type})]" if chart_type else "[CHART]"
            lines.append(header)
            if takeaway:
                lines.append(f"  Insight: {takeaway}")
            if trends:
                lines.append(f"  Trends: {trends}")
            if data_table:
                lines.append(data_table)

        elif btype == BlockType.CAPTION:
            text = (p.get("text") or "").strip()
            if text:
                lines.append(f"_{text}_")
            # Emit formulas referenced from this caption
            for fid in sorted(
                block.relations.get("embedded_formulas", []),
                key=lambda fid: (
                    block_map[fid].bbox.y1 if fid in block_map else 0,
                    block_map[fid].bbox.x1 if fid in block_map else 0,
                ),
            ):
                child = block_map.get(fid)
                if not child:
                    continue
                fp = child.payload or {}
                latex = (fp.get("latex") or "").strip()
                meaning = (fp.get("meaning") or "").strip()
                if latex:
                    lines.append(f"  {{formula: {latex}}}")
                    if meaning:
                        lines.append(f"  [meaning: {meaning}]")

    return "\n".join(lines)


def node_aggregate(state: AgentState) -> AgentState:
    trace = state.get("_trace") or {}
    entry = node_start(trace, "aggregate")
    blocks = state.get("blocks", [])
    ordered = sorted(
        blocks,
        key=lambda b: (b.page_index, b.reading_order if b.reading_order >= 0 else 999_999),
    )
    type_counts: Dict[str, int] = {}
    for b in ordered:
        type_counts[b.block_type.value] = type_counts.get(b.block_type.value, 0) + 1

    full_text = _build_full_text(ordered)

    # Reading order manifest: (block_id, page, type, short_preview)
    reading_order = [
        {
            "block_id":     b.block_id,
            "page":         b.page_index + 1,
            "reading_order": b.reading_order,
            "type":         b.block_type.value,
            "bbox":         {"x1": b.bbox.x1, "y1": b.bbox.y1, "x2": b.bbox.x2, "y2": b.bbox.y2},
        }
        for b in ordered
        if b.parent_id is None
        and b.detector_label != "aside_text"
        and b.block_type not in {BlockType.HEADER, BlockType.FOOTER, BlockType.OTHER}
        and not (
            b.block_type == BlockType.TEXT
            and len((b.payload.get("text") or "").strip()) < 3
        )
    ]

    node_end(entry)
    finish_trace(trace)

    from .tracer import compute_metrics
    metrics = compute_metrics(trace)

    logger.info("[aggregate] done - %d blocks, %.0f tokens, $%.4f, total %.2fs",
                len(ordered),
                metrics.get("total_tokens", 0),
                metrics.get("total_cost_usd", 0),
                trace.get("total_duration_s", 0))

    output: Dict = {
        "run_id":     state.get("run_id"),
        "input_path": state.get("input_path"),
        "status":     "done",
        "warnings":   state.get("warnings", []),
        "pages":      state.get("page_sizes", []),
        "blocks":     [b.as_dict() for b in ordered],
        "full_text":  full_text,
        "reading_order": reading_order,
        "metrics":    metrics,
        "summary": {
            "num_pages":        len(state.get("page_sizes", [])),
            "num_blocks":       len(ordered),
            "types":            type_counts,
            "full_text_chars":  len(full_text),
            "total_duration_s": metrics.get("total_duration_s"),
            "total_tokens":     metrics.get("total_tokens"),
            "total_cost_usd":   metrics.get("total_cost_usd"),
            "num_llm_calls":    metrics.get("num_llm_calls"),
        },
    }
    return {**state, "output": output, "status": "done", "_trace": trace}
