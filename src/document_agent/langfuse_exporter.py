"""
Langfuse exporter for the DocumentAgent pipeline tracer.

Reads the internal `_trace` dict produced by tracer.py and posts it to
Langfuse as a structured trace with spans (pipeline nodes) and generations
(LLM API calls).

Activation
----------
Set three env vars (add to .env):

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL

If LANGFUSE_PUBLIC_KEY is not set the exporter is a silent no-op.

Langfuse trace structure
------------------------
Trace (one per pipeline run)
  ├── Span: load_document
  ├── Span: detect_layout
  ├── Span: reading_order
  ├── Span: hierarchy
  ├── Span: text_specialist
  ├── Span: formula_specialist   (contains Generations below)
  │     ├── Generation: formula b45  (gpt-4.1-mini)
  │     ├── Generation: formula b46
  │     └── …
  ├── Span: table_specialist
  │     └── Generation: table b12
  ├── …
  └── Span: aggregate
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _ts(t: float) -> datetime:
    """Unix timestamp → UTC datetime for Langfuse."""
    return datetime.fromtimestamp(t, tz=timezone.utc)


def _get_client():
    """Return a Langfuse client or None if not configured / not installed."""
    public_key  = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except ImportError:
        return None
    except Exception:
        return None


def export(
    trace_data: Dict[str, Any],
    output: Dict[str, Any],
    input_path: str,
    run_id: str,
) -> bool:
    """
    Export a finished pipeline trace to Langfuse.

    Parameters
    ----------
    trace_data : _trace dict from the pipeline (nodes + blocks + llm_calls)
    output     : structured output dict (summary, metrics, warnings)
    input_path : original document path (shown in Langfuse UI)
    run_id     : UUID for this run — used as the Langfuse trace ID so
                 re-exporting the same run is idempotent.

    Returns True if export succeeded, False otherwise (never raises).
    """
    lf = _get_client()
    if lf is None:
        return False

    try:
        summary = output.get("summary", {})
        warnings = output.get("warnings", [])
        metrics  = output.get("metrics", {})

        # ── Trace ─────────────────────────────────────────────────────────────
        pipeline_start = trace_data.get("pipeline_start", time.time())
        pipeline_end   = trace_data.get("pipeline_end")   or time.time()

        lf_trace = lf.trace(
            id=run_id,
            name="document_pipeline",
            input={"input_path": os.path.basename(input_path)},
            output={
                "status":     output.get("status"),
                "num_blocks": summary.get("num_blocks"),
                "full_text_chars": summary.get("full_text_chars"),
            },
            metadata={
                "input_path":       input_path,
                "num_pages":        summary.get("num_pages"),
                "num_blocks":       summary.get("num_blocks"),
                "block_types":      summary.get("types"),
                "total_cost_usd":   summary.get("total_cost_usd"),
                "total_tokens":     summary.get("total_tokens"),
                "num_llm_calls":    summary.get("num_llm_calls"),
                "total_duration_s": summary.get("total_duration_s"),
                "engine_dist":      metrics.get("engine_distribution"),
                "warnings":         warnings[:20],  # cap to avoid huge payloads
            },
            tags=["document-agent"],
            start_time=_ts(pipeline_start),
            end_time=_ts(pipeline_end),
        )

        # ── Build span map: node_name → langfuse span ─────────────────────────
        # Multiple nodes can share the same name (parallel specialists).
        # We create one span per entry in trace_data["nodes"] and key by name
        # for the generation-parent lookup below.
        span_by_name: Dict[str, Any] = {}

        for node_entry in trace_data.get("nodes", []):
            node_name = node_entry.get("node", "unknown")
            start     = node_entry.get("start")
            end       = node_entry.get("end")
            if not start:
                continue

            span = lf_trace.span(
                name=node_name,
                start_time=_ts(start),
                end_time=_ts(end) if end else _ts(pipeline_end),
                metadata={
                    "duration_s": node_entry.get("duration_s"),
                },
            )
            # Keep first span per node name for generation parenting
            span_by_name.setdefault(node_name, span)

        # ── Block summary as a single span metadata (avoids N×spans for blocks)
        # Individual block details live in the summary.json; here we just attach
        # per-type counts so the Langfuse UI shows them.
        lf_trace.span(
            name="blocks_summary",
            start_time=_ts(pipeline_start),
            end_time=_ts(pipeline_end),
            metadata={
                "type_distribution": metrics.get("type_distribution"),
                "blocks_per_page":   metrics.get("blocks_per_page"),
                "slowest_blocks":    metrics.get("slowest_blocks", [])[:5],
            },
        )

        # ── Generations (one per LLM call) ────────────────────────────────────
        # Nest under the matching specialist span when available.
        _specialist_to_node = {
            "text":    "text_specialist",
            "formula": "formula_specialist",
            "image":   "image_specialist",
            "chart":   "chart_specialist",
            "table":   "table_specialist",
            "mixed":   "mixed_specialist",
            "other":   "other_specialist",
        }

        for call in trace_data.get("llm_calls", []):
            ts_end   = call.get("ts", time.time())
            ts_start = ts_end - call.get("duration_s", 0)
            specialist = call.get("specialist", "unknown")
            node_name  = _specialist_to_node.get(specialist, specialist + "_specialist")
            parent     = span_by_name.get(node_name, lf_trace)

            parent.generation(
                name=f"{specialist}/{call.get('block_id', '?')}",
                model=call.get("model", "unknown"),
                model_parameters={"temperature": 0},
                input=call.get("prompt_text", ""),
                output=call.get("response_text", ""),
                usage={
                    "input":  call.get("prompt_tokens", 0),
                    "output": call.get("completion_tokens", 0),
                    "total":  call.get("total_tokens", 0),
                    "unit":   "TOKENS",
                },
                start_time=_ts(ts_start),
                end_time=_ts(ts_end),
                metadata={
                    "block_id":   call.get("block_id"),
                    "page_index": call.get("page_index"),
                    "block_type": call.get("block_type"),
                    "prompt_name": call.get("prompt_name"),
                    "cost_usd":   call.get("cost_usd"),
                },
            )

        lf.flush()
        return True

    except Exception:
        return False
