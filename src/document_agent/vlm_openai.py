from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional, Tuple

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
- title: table title/caption if visible inside the crop
- headers: list of column header strings (in order)
- rows: list of row objects where each object maps header → cell value (use null for empty cells)
- notes: any footnotes or annotations below the table
- summary: one-sentence description of what the table shows
- confidence: float 0-1"""


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
# Core call
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


def _run_openai_vision(prompt: str, model: str, img, api_key: str) -> str:
    from openai import OpenAI

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
    return _extract_content_text(resp.choices[0].message.content).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

def analyze_image(img, cfg: OpenAIVLMConfig) -> Tuple[str, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return _run_openai_vision(IMAGE_PROMPT, cfg.image_model, img, cfg.api_key), cfg.image_model


def analyze_chart(img, cfg: OpenAIVLMConfig) -> Tuple[str, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return _run_openai_vision(CHART_PROMPT, cfg.chart_model, img, cfg.api_key), cfg.chart_model


def analyze_formula(img, cfg: OpenAIVLMConfig) -> Tuple[str, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return _run_openai_vision(FORMULA_PROMPT, cfg.formula_model, img, cfg.api_key), cfg.formula_model


def analyze_mixed(img, cfg: OpenAIVLMConfig) -> Tuple[str, str]:
    """Analyze a MIXED block (inline math embedded in prose text)."""
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return _run_openai_vision(MIXED_PROMPT, cfg.mixed_model, img, cfg.api_key), cfg.mixed_model


def analyze_table(img, cfg: OpenAIVLMConfig) -> Tuple[str, str]:
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return _run_openai_vision(TABLE_PROMPT, cfg.table_model, img, cfg.api_key), cfg.table_model
