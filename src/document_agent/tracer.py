"""
Pipeline tracer — timing + structured event log + LLM call audit.

Schema
------
_trace = {
    "nodes": [
        {"node": "detect_layout", "start": ..., "end": ..., "duration_s": 1.23},
        ...
    ],
    "blocks": [
        {
            "block_id": "b1",
            "page_index": 0,
            "type": "text",
            "detector_label": "paragraph",
            "confidence": 0.97,
            "bbox": {"x1":…},
            "reading_order": 3,
            "specialist": "text",
            "engine": "pytesseract",
            "text_preview": "The quick brown…",
            "duration_s": 0.04,
        },
        ...
    ],
    "llm_calls": [
        {
            "block_id":        "b45",
            "page_index":      2,
            "block_type":      "formula",
            "specialist":      "formula",
            "model":           "gpt-4.1-mini",
            "prompt_name":     "FORMULA_PROMPT",
            "prompt_text":     "You are analyzing…",
            "response_text":   "{\"latex\": …}",
            "prompt_tokens":   832,
            "completion_tokens": 120,
            "total_tokens":    952,
            "cost_usd":        0.000534,
            "duration_s":      3.21,
            "ts":              1713012345.67,
        },
        ...
    ],
    "pipeline_start":        1713000000.0,
    "pipeline_end":          None,
    "total_duration_s":      None,
}
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Tracer helpers
# ──────────────────────────────────────────────────────────────────────────────

def init_trace() -> Dict[str, Any]:
    return {
        "nodes": [],
        "blocks": [],
        "llm_calls": [],
        "pipeline_start": time.time(),
        "pipeline_end": None,
        "total_duration_s": None,
    }


def node_start(trace: Dict, node_name: str) -> Dict:
    """Record node start; returns an entry to be passed to node_end."""
    trace.setdefault("nodes", [])
    trace.setdefault("blocks", [])
    trace.setdefault("llm_calls", [])
    trace.setdefault("pipeline_start", time.time())
    trace.setdefault("pipeline_end", None)
    trace.setdefault("total_duration_s", None)
    entry = {"node": node_name, "start": time.time(), "end": None, "duration_s": None}
    trace["nodes"].append(entry)
    return entry


def node_end(entry: Dict) -> None:
    entry["end"] = time.time()
    entry["duration_s"] = round(entry["end"] - entry["start"], 4)


def record_block(
    trace: Dict,
    block,
    specialist: str,
    engine: str,
    duration_s: float,
    payload: Dict,
) -> None:
    """Log one processed block into the trace."""
    text = (
        payload.get("text")           # OCR text
        or payload.get("full_text")   # mixed block
        or payload.get("latex")       # formula (flat field)
        or payload.get("summary")     # image / table / chart
        or payload.get("takeaway")    # chart insight
        or payload.get("meaning")     # formula meaning
        or ""
    )
    preview = str(text)[:120].replace("\n", " ").strip()

    trace.setdefault("blocks", []).append({
        "block_id":       block.block_id,
        "page_index":     block.page_index,
        "type":           block.block_type.value,
        "detector_label": block.detector_label,
        "confidence":     round(float(block.confidence), 4),
        "reading_order":  block.reading_order,
        "bbox": {
            "x1": block.bbox.x1, "y1": block.bbox.y1,
            "x2": block.bbox.x2, "y2": block.bbox.y2,
        },
        "specialist":     specialist,
        "engine":         engine,
        "text_preview":   preview,
        "duration_s":     round(duration_s, 4),
    })


def record_llm_call(
    trace: Dict,
    block_id: str,
    page_index: int,
    block_type: str,
    specialist: str,
    model: str,
    prompt_name: str,
    prompt_text: str,
    response_text: str,
    usage: Dict,
    duration_s: float,
) -> None:
    """Append a single LLM API call record to the trace.

    Thread-safe via Python GIL (list.append is atomic).
    Token totals are computed lazily in compute_metrics to avoid
    read-modify-write races across parallel specialist threads.
    """
    trace.setdefault("llm_calls", []).append({
        "block_id":          block_id,
        "page_index":        page_index,
        "block_type":        block_type,
        "specialist":        specialist,
        "model":             model,
        "prompt_name":       prompt_name,
        "prompt_text":       prompt_text,
        "response_text":     response_text,
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens", 0),
        "cost_usd":          usage.get("cost_usd", 0.0),
        "duration_s":        round(duration_s, 4),
        "ts":                time.time(),
    })


def finish_trace(trace: Dict) -> None:
    trace["pipeline_end"] = time.time()
    trace["total_duration_s"] = round(
        trace["pipeline_end"] - trace["pipeline_start"], 4
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics summary derived from trace
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(trace: Dict) -> Dict[str, Any]:
    """Derive a metrics dict from a finished trace."""
    nodes  = trace.get("nodes", [])
    blocks = trace.get("blocks", [])
    calls  = trace.get("llm_calls", [])

    # Per-node timing
    node_timing: Dict[str, float] = {}
    for e in nodes:
        name = e["node"]
        dur  = e.get("duration_s") or 0.0
        # Accumulate (parallel nodes with same name are summed)
        node_timing[name] = round(node_timing.get(name, 0.0) + dur, 4)

    # Block type distribution
    type_counts: Dict[str, int] = {}
    for b in blocks:
        type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1

    # Engine usage
    engine_counts: Dict[str, int] = {}
    for b in blocks:
        eng = b.get("engine", "unknown")
        engine_counts[eng] = engine_counts.get(eng, 0) + 1

    # Slowest blocks
    slowest = sorted(blocks, key=lambda x: x.get("duration_s", 0), reverse=True)[:10]

    # Per-page block counts
    page_counts: Dict[int, int] = {}
    for b in blocks:
        p = b["page_index"]
        page_counts[p] = page_counts.get(p, 0) + 1

    # Token totals across all LLM calls
    total_prompt_tokens     = sum(c.get("prompt_tokens", 0)     for c in calls)
    total_completion_tokens = sum(c.get("completion_tokens", 0) for c in calls)
    total_tokens            = sum(c.get("total_tokens", 0)      for c in calls)
    total_cost_usd          = round(sum(c.get("cost_usd", 0.0)  for c in calls), 6)

    # Per-model breakdown
    model_breakdown: Dict[str, Dict] = {}
    for c in calls:
        m = c.get("model", "unknown")
        if m not in model_breakdown:
            model_breakdown[m] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        model_breakdown[m]["calls"]             += 1
        model_breakdown[m]["prompt_tokens"]     += c.get("prompt_tokens", 0)
        model_breakdown[m]["completion_tokens"] += c.get("completion_tokens", 0)
        model_breakdown[m]["cost_usd"]          = round(model_breakdown[m]["cost_usd"] + c.get("cost_usd", 0.0), 6)

    return {
        "total_duration_s":        trace.get("total_duration_s"),
        "num_blocks_traced":       len(blocks),
        "num_llm_calls":           len(calls),
        "total_prompt_tokens":     total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens":            total_tokens,
        "total_cost_usd":          total_cost_usd,
        "model_breakdown":         model_breakdown,
        "node_timing_s":           node_timing,
        "type_distribution":       type_counts,
        "engine_distribution":     engine_counts,
        "blocks_per_page":         page_counts,
        "slowest_blocks": [
            {k: v for k, v in b.items() if k != "text_preview"}
            for b in slowest
        ],
    }
