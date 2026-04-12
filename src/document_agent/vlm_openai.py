from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_PROMPT = """You are analyzing a region cropped from a technical document (paper, report, slide).
Describe the image precisely and return ONLY valid JSON with these keys:
- summary: one-sentence description of what the image shows
- key_elements: list of the main visual elements (objects, labels, arrows, annotations)
- visible_text: any text legible inside the image (exact transcription)
- spatial_layout: description of the spatial arrangement of elements
- confidence: float 0-1 reflecting your certainty"""

CHART_PROMPT = """You are analyzing a chart/graph/plot cropped from a technical document.
Return ONLY valid JSON with these keys:
- chart_type: e.g. bar, line, scatter, pie, heatmap, box
- title: chart title if visible
- axes: {x_label, x_unit, y_label, y_unit, x_range, y_range}
- series: list of series names / legend entries
- trends: key trends observed (monotonic increase, peak at X, etc.)
- extrema: notable minimum and maximum values with their coordinates
- comparisons: relative comparisons between series or categories
- data_table: if readable, a markdown table of approximate data points
- takeaway: one-sentence main insight
- confidence: float 0-1"""

FORMULA_PROMPT = """You are analyzing a mathematical formula or equation cropped from a technical document.
Return ONLY valid JSON with these keys:
- latex: the full LaTeX representation of the formula (use \\\\[ \\\\] for display, $ $ for inline)
- symbols: list of {symbol, meaning} for each variable / operator
- formula_type: e.g. definition, theorem, constraint, loss_function, update_rule
- meaning: plain-English explanation of what the formula expresses
- confidence: float 0-1"""

MIXED_PROMPT = """You are analyzing a region of a technical document that contains BOTH prose text and an embedded mathematical formula.
The formula appears inline within the surrounding text.
Return ONLY valid JSON with these keys:
- full_text: the complete reconstructed text with the formula rendered as inline LaTeX (use $...$ notation)
- formula_latex: just the LaTeX of the embedded formula(s)
- plain_text: the prose text without the formula (surrounding context only)
- confidence: float 0-1"""

TABLE_PROMPT = """You are analyzing a table cropped from a technical document.
Return ONLY valid JSON with these keys:
- title: table title/caption if visible inside the crop (empty string if none)
- headers: list of column header strings in left-to-right order.
  For multi-level headers (e.g. a top header spanning sub-columns), flatten them as "Parent / Child".
  For row-header tables (first column labels rows), include the row-header column name as the first header.
- rows: list of row objects where each key is a header string and each value is the cell content.
  Use null for empty/merged cells. For merged cells that span multiple rows, repeat the value in each row.
  Preserve numeric formatting exactly (units, percentages, scientific notation).
- row_headers: if the first column contains row labels (not data), list those labels separately here
  so downstream code can distinguish them from numeric data columns. Otherwise empty list.
- notes: any footnotes, annotations, or legends below the table (empty string if none)
- summary: one-sentence description of what the table shows and what its main finding is
- confidence: float 0-1 reflecting how clearly you could read the table"""

# Map prompt name → prompt text (for traceability)
PROMPT_REGISTRY: Dict[str, str] = {
    "IMAGE_PROMPT": IMAGE_PROMPT,
    "CHART_PROMPT": CHART_PROMPT,
    "FORMULA_PROMPT": FORMULA_PROMPT,
    "MIXED_PROMPT": MIXED_PROMPT,
    "TABLE_PROMPT": TABLE_PROMPT,
}


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenAIVLMConfig:
    api_key: Optional[str]
    image_model: str = "gpt-4.1-mini"
    chart_model: str = "gpt-4.1-mini"
    formula_model: str = "gpt-4.1-mini"
    mixed_model: str = "gpt-4.1-mini"
    table_model: str = "gpt-4.1-mini"


def load_vlm_config() -> OpenAIVLMConfig:
    return OpenAIVLMConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        image_model=os.getenv("OPENAI_VLM_IMAGE_MODEL", "gpt-4.1-mini"),
        chart_model=os.getenv("OPENAI_VLM_CHART_MODEL", "gpt-4.1-mini"),
        formula_model=os.getenv("OPENAI_VLM_FORMULA_MODEL", "gpt-4.1-mini"),
        mixed_model=os.getenv("OPENAI_VLM_MIXED_MODEL", "gpt-4.1-mini"),
        table_model=os.getenv("OPENAI_VLM_TABLE_MODEL", "gpt-4.1-mini"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Token cost table  (USD per 1M tokens, gpt-4.1-mini as of 2025-04)
# ──────────────────────────────────────────────────────────────────────────────

_COST_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini":   {"input": 0.40,  "output": 1.60},
    "gpt-4.1":        {"input": 2.00,  "output": 8.00},
    "gpt-4o":         {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
}

def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _COST_PER_1M.get(model, _COST_PER_1M["gpt-4.1-mini"])
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


# ──────────────────────────────────────────────────────────────────────────────
# Core call  — returns (response_text, usage_dict, prompt_text, call_duration_s)
# ──────────────────────────────────────────────────────────────────────────────

def _img_to_data_url(img) -> str:
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode crop as PNG.")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_content_text(message_content) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        texts = [
            item.get("text", "")
            for item in message_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(t for t in texts if t)
    return ""


def _run_openai_vision(
    prompt: str,
    model: str,
    img,
    api_key: str,
) -> Tuple[str, Dict, float]:
    """Call the OpenAI vision API.

    Returns
    -------
    text : str
        Raw JSON string from the model.
    usage : dict
        {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N, "cost_usd": F}
    duration_s : float
        Wall-clock time for the API call.
    """
    from openai import OpenAI

    t0 = time.time()
    data_url = _img_to_data_url(img)
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return concise, valid JSON only. Do not wrap in markdown code fences."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )
    duration_s = time.time() - t0

    u = resp.usage
    pt = u.prompt_tokens if u else 0
    ct = u.completion_tokens if u else 0
    usage = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "cost_usd": round(estimate_cost_usd(model, pt, ct), 6),
    }
    text = _extract_content_text(resp.choices[0].message.content).strip()
    return text, usage, duration_s


# ──────────────────────────────────────────────────────────────────────────────
# Batch call — sends N images in one request, returns N parsed JSON dicts
# ──────────────────────────────────────────────────────────────────────────────

def _run_openai_vision_batch(
    prompt: str,
    model: str,
    imgs: list,
    api_key: str,
) -> Tuple[list, Dict, float]:
    """Send multiple images in a single OpenAI call.

    The model is asked to return ``{"results": [...]}`` with one entry per
    image in order.  Falls back to an empty list on any failure so callers
    can retry individually.

    Returns
    -------
    results : list[dict]
        Parsed dicts, one per image.  May be shorter than ``imgs`` on error.
    usage : dict
    duration_s : float
    """
    from openai import OpenAI
    import json as _json

    n = len(imgs)
    t0 = time.time()

    content: list = [
        {
            "type": "text",
            "text": (
                f"You will receive {n} image(s) numbered 1 to {n}. "
                f"For EACH image apply the following instructions and return a JSON object "
                f"with a single key 'results' containing a list of {n} objects in order.\n\n"
                f"{prompt}"
            ),
        }
    ]
    for idx, img in enumerate(imgs):
        content.append({"type": "text", "text": f"Image {idx + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(img), "detail": "high"}})

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return concise, valid JSON only. Do not wrap in markdown code fences."},
                {"role": "user", "content": content},
            ],
        )
        duration_s = time.time() - t0
        u = resp.usage
        pt = u.prompt_tokens if u else 0
        ct = u.completion_tokens if u else 0
        usage = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "cost_usd": round(estimate_cost_usd(model, pt, ct), 6),
        }
        raw = _extract_content_text(resp.choices[0].message.content).strip()
        parsed = _json.loads(raw)
        results = parsed.get("results", [])
        if not isinstance(results, list):
            results = []
        return results, usage, duration_s
    except Exception:
        duration_s = time.time() - t0
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}, duration_s


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# Each returns (response_text, model, usage_dict, prompt_name)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_image(img, cfg: OpenAIVLMConfig) -> Tuple[str, str, Dict, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    text, usage, _ = _run_openai_vision(IMAGE_PROMPT, cfg.image_model, img, cfg.api_key)
    return text, cfg.image_model, usage, "IMAGE_PROMPT"


def analyze_chart(img, cfg: OpenAIVLMConfig) -> Tuple[str, str, Dict, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    text, usage, _ = _run_openai_vision(CHART_PROMPT, cfg.chart_model, img, cfg.api_key)
    return text, cfg.chart_model, usage, "CHART_PROMPT"


def analyze_formula(img, cfg: OpenAIVLMConfig) -> Tuple[str, str, Dict, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    text, usage, _ = _run_openai_vision(FORMULA_PROMPT, cfg.formula_model, img, cfg.api_key)
    return text, cfg.formula_model, usage, "FORMULA_PROMPT"


def analyze_mixed(img, cfg: OpenAIVLMConfig) -> Tuple[str, str, Dict, str]:
    """Analyze a MIXED block (inline math embedded in prose text)."""
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    text, usage, _ = _run_openai_vision(MIXED_PROMPT, cfg.mixed_model, img, cfg.api_key)
    return text, cfg.mixed_model, usage, "MIXED_PROMPT"


def analyze_table(img, cfg: OpenAIVLMConfig) -> Tuple[str, str, Dict, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    text, usage, _ = _run_openai_vision(TABLE_PROMPT, cfg.table_model, img, cfg.api_key)
    return text, cfg.table_model, usage, "TABLE_PROMPT"
