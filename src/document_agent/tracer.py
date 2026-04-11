"""
Pipeline tracer — timing + structured event log.

Every pipeline node calls ``Tracer.start_node`` / ``Tracer.end_node``.
Every specialist block call logs a ``block_event``.

The tracer lives in ``AgentState["_trace"]`` as a plain dict so it survives
LangGraph serialisation without custom reducers.

Schema
------
_trace = {
    "nodes": [
        {"node": "detect_layout", "start": 1713000000.0, "end": ..., "duration_s": 1.23},
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
            "specialist": "text",
            "engine": "paddleocr",
            "text_preview": "The quick brown…",
            "duration_s": 0.04,
        },
        ...
    ],
    "pipeline_start": 1713000000.0,
    "pipeline_end": None,
    "total_duration_s": None,
}
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Tracer helpers  (operate on the plain _trace dict stored in state)
# ──────────────────────────────────────────────────────────────────────────────

def init_trace() -> Dict[str, Any]:
    return {
        "nodes": [],
        "blocks": [],
        "pipeline_start": time.time(),
        "pipeline_end": None,
        "total_duration_s": None,
    }


def node_start(trace: Dict, node_name: str) -> Dict:
    """Record node start; returns an entry to be passed to node_end.

    Safe to call even when *trace* is a bare ``{}`` (e.g. in unit tests that
    do not initialise the tracer via ``node_load_document``).
    """
    trace.setdefault("nodes", [])
    trace.setdefault("blocks", [])
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
    # Extract a short text preview from the payload
    text = (
        payload.get("text")
        or payload.get("full_text")
        or payload.get("formula_latex")
        or payload.get("description")
        or ""
    )
    preview = str(text)[:120].replace("\n", " ").strip()

    trace["blocks"].append({
        "block_id": block.block_id,
        "page_index": block.page_index,
        "type": block.block_type.value,
        "detector_label": block.detector_label,
        "confidence": round(float(block.confidence), 4),
        "reading_order": block.reading_order,
        "bbox": {
            "x1": block.bbox.x1, "y1": block.bbox.y1,
            "x2": block.bbox.x2, "y2": block.bbox.y2,
        },
        "specialist": specialist,
        "engine": engine,
        "text_preview": preview,
        "duration_s": round(duration_s, 4),
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
    nodes = trace.get("nodes", [])
    blocks = trace.get("blocks", [])

    # Per-node timing
    node_timing = {e["node"]: e.get("duration_s", 0) for e in nodes}

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

    return {
        "total_duration_s": trace.get("total_duration_s"),
        "num_blocks_traced": len(blocks),
        "node_timing_s": node_timing,
        "type_distribution": type_counts,
        "engine_distribution": engine_counts,
        "blocks_per_page": page_counts,
        "slowest_blocks": [
            {k: v for k, v in b.items() if k != "text_preview"}
            for b in slowest
        ],
    }
