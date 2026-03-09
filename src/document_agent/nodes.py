from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import cv2

from .hierarchy import build_hierarchy
from .io import load_document_pages
from .layout import detect_layout_blocks
from .order import apply_reading_order
from .types import AgentState, BlockType
from .vlm_openai import analyze_chart, analyze_formula, analyze_image, load_vlm_config


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


def node_detect_layout(state: AgentState) -> AgentState:
    warnings = list(state.get("warnings", []))
    blocks = detect_layout_blocks(state["pages"], warnings)
    return {**state, "blocks": blocks, "warnings": warnings}


def node_reading_order(state: AgentState) -> AgentState:
    blocks = apply_reading_order(state.get("blocks", []))
    return {**state, "blocks": blocks}


def node_hierarchy(state: AgentState) -> AgentState:
    blocks = build_hierarchy(state.get("blocks", []))
    return {**state, "blocks": blocks}


def _crop_block_image(state: AgentState, block_id: str):
    pages = state["pages"]
    block = next(b for b in state["blocks"] if b.block_id == block_id)
    img = pages[block.page_index]
    return img[block.bbox.y1 : block.bbox.y2, block.bbox.x1 : block.bbox.x2]


def _extract_text_payload(crop):
    if crop is None or not crop.size:
        return {"text": "", "ocr_engine": "pytesseract"}
    try:
        import pytesseract

        text = pytesseract.image_to_string(crop)
        return {"text": text.strip(), "ocr_engine": "pytesseract"}
    except Exception as exc:  # pragma: no cover
        return {"text": "", "ocr_engine": "fallback", "ocr_error": str(exc)}


def _process_block_task(task: Dict):
    block = task["block"]
    crop = task["crop"]
    cfg = task["cfg"]
    kind = task["kind"]
    payload = {}
    warnings: List[str] = []

    if kind == "text":
        payload = _extract_text_payload(crop)
    elif kind == "image":
        h, w = crop.shape[:2] if crop is not None and crop.size else (0, 0)
        if crop is None or not crop.size:
            payload = {"description": "", "vlm_engine": "openai", "vlm_model": cfg.image_model}
        else:
            try:
                text, model_name = analyze_image(crop, cfg)
                payload = {"description": text, "vlm_engine": "openai", "vlm_model": model_name}
            except Exception as exc:  # pragma: no cover
                payload = {
                    "description": f"Image region {w}x{h}. OpenAI VLM unavailable.",
                    "vlm_engine": "fallback",
                    "vlm_model": "",
                }
                warnings.append(f"image_node fallback for {block.block_id}: {exc}")
    elif kind == "chart":
        if crop is None or not crop.size:
            payload = {
                "chart_semantics": {"status": "empty_crop", "data": ""},
                "chart_engine": "openai",
                "chart_model": cfg.chart_model,
            }
        else:
            try:
                text, model_name = analyze_chart(crop, cfg)
                payload = {
                    "chart_semantics": {"status": "ok", "data": text},
                    "chart_engine": "openai",
                    "chart_model": model_name,
                }
            except Exception as exc:  # pragma: no cover
                payload = {
                    "chart_semantics": {"status": "fallback", "data": ""},
                    "chart_engine": "fallback",
                    "chart_model": "",
                }
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
        except Exception as exc:  # pragma: no cover
            payload = {
                "formula_latex": "",
                "formula_engine": "fallback",
                "formula_model": "",
                "preprocess": {"shape": tuple(gray.shape) if gray is not None else (0, 0)},
            }
            warnings.append(f"formula_node fallback for {block.block_id}: {exc}")

    return {"block_id": block.block_id, "payload": payload, "warnings": warnings}


def _run_specialist_kind(state: AgentState, kind: str):
    blocks = state.get("blocks", [])
    cfg = load_vlm_config()
    tasks = []
    for block in blocks:
        if kind == "text" and block.block_type != BlockType.TEXT:
            continue
        if kind == "image" and block.block_type not in {BlockType.IMAGE, BlockType.FIGURE}:
            continue
        if kind == "chart" and block.block_type != BlockType.CHART:
            continue
        if kind == "formula" and block.block_type != BlockType.FORMULA:
            continue
        if kind == "other" and block.block_type in {
            BlockType.TEXT,
            BlockType.IMAGE,
            BlockType.FIGURE,
            BlockType.CHART,
            BlockType.FORMULA,
        }:
            continue
        tasks.append({"kind": kind, "block": block, "crop": _crop_block_image(state, block.block_id), "cfg": cfg})

    if not tasks:
        return {}, []

    max_workers = int(os.getenv("DOCAGENT_MAX_WORKERS", "4"))
    max_workers = max(1, min(max_workers, len(tasks)))
    updates: Dict[str, Dict] = {}
    warnings: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_block_task, t) for t in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            updates[result["block_id"]] = result["payload"]
            warnings.extend(result["warnings"])

    return updates, warnings


def node_text_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "text")
    return {**state, "text_updates": updates, "text_warnings": warns}


def node_image_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "image")
    return {**state, "image_updates": updates, "image_warnings": warns}


def node_chart_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "chart")
    return {**state, "chart_updates": updates, "chart_warnings": warns}


def node_formula_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "formula")
    return {**state, "formula_updates": updates, "formula_warnings": warns}


def node_other_specialist(state: AgentState) -> AgentState:
    updates, warns = _run_specialist_kind(state, "other")
    return {**state, "other_updates": updates, "other_warnings": warns}


def node_reduce_specialists(state: AgentState) -> AgentState:
    blocks = state.get("blocks", [])
    by_id = {b.block_id: b for b in blocks}
    warnings = list(state.get("warnings", []))

    for key in ["text_updates", "image_updates", "chart_updates", "formula_updates", "other_updates"]:
        updates = state.get(key, {}) or {}
        for block_id, payload in updates.items():
            if block_id in by_id:
                by_id[block_id].payload.update(payload)

    for key in ["text_warnings", "image_warnings", "chart_warnings", "formula_warnings", "other_warnings"]:
        warnings.extend(state.get(key, []) or [])

    return {**state, "blocks": blocks, "warnings": warnings}


def node_association(state: AgentState) -> AgentState:
    blocks = state.get("blocks", [])
    captions = [b for b in blocks if b.block_type == BlockType.CAPTION]
    figure_like = [b for b in blocks if b.block_type in {BlockType.FIGURE, BlockType.IMAGE, BlockType.CHART}]

    for cap in captions:
        same_page = [f for f in figure_like if f.page_index == cap.page_index]
        if not same_page:
            continue
        nearest = min(
            same_page,
            key=lambda fig: abs(fig.bbox.cy - cap.bbox.cy) + abs(fig.bbox.cx - cap.bbox.cx),
        )
        cap.relations["describes"] = nearest.block_id
        nearest.relations.setdefault("captions", []).append(cap.block_id)

    # Case: full-page image/figure/chart regions.
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

    # Case: formula embedded inside text blocks.
    id_to_block = {b.block_id: b for b in blocks}
    formulas = [b for b in blocks if b.block_type == BlockType.FORMULA]
    for formula in formulas:
        if formula.parent_id and formula.parent_id in id_to_block:
            parent = id_to_block[formula.parent_id]
            if parent.block_type == BlockType.TEXT:
                parent.relations.setdefault("embedded_formulas", []).append(formula.block_id)
                formula.relations["embedded_in_text"] = parent.block_id

    return {**state, "blocks": blocks}


def node_aggregate(state: AgentState) -> AgentState:
    blocks = state.get("blocks", [])
    ordered = sorted(blocks, key=lambda b: (b.page_index, b.reading_order if b.reading_order >= 0 else 999999))
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
            "types": {},
        },
    }
    for b in ordered:
        output["summary"]["types"][b.block_type.value] = output["summary"]["types"].get(b.block_type.value, 0) + 1

    return {**state, "output": output, "status": "done"}
