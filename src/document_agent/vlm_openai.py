from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


IMAGE_PROMPT = (
    "You are analyzing a technical document region.\n"
    "Describe the image objectively and return JSON with keys: "
    "summary, key_elements, visible_text, confidence."
)

CHART_PROMPT = (
    "Analyze this chart region from a technical document.\n"
    "Return JSON with keys: chart_type, axes, trends, extrema, comparisons, takeaway, confidence."
)

FORMULA_PROMPT = (
    "Extract the mathematical expression from this formula region.\n"
    "Return JSON with keys: latex, symbols, meaning, confidence."
)


@dataclass
class OpenAIVLMConfig:
    api_key: Optional[str]
    image_model: str = "gpt-4.1-mini"
    chart_model: str = "gpt-4.1-mini"
    formula_model: str = "gpt-4.1-mini"


def load_vlm_config() -> OpenAIVLMConfig:
    return OpenAIVLMConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        image_model=os.getenv("OPENAI_VLM_IMAGE_MODEL", "gpt-4.1-mini"),
        chart_model=os.getenv("OPENAI_VLM_CHART_MODEL", "gpt-4.1-mini"),
        formula_model=os.getenv("OPENAI_VLM_FORMULA_MODEL", "gpt-4.1-mini"),
    )


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
        texts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join([t for t in texts if t])
    return ""


def _run_openai_vision(prompt: str, model: str, img, api_key: str) -> str:
    from openai import OpenAI

    data_url = _img_to_data_url(img)
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "Return concise, valid JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return _extract_content_text(resp.choices[0].message.content).strip()


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

