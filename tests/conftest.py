"""
Shared pytest fixtures for DocumentAgent tests.
All fixtures are pure in-memory — no disk I/O, no external models.
"""
from __future__ import annotations

import numpy as np
import pytest

from document_agent.types import BlockType, BoundingBox, DocumentBlock


# ──────────────────────────────────────────────────────────────────────────────
# BoundingBox helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_bbox(x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


# ──────────────────────────────────────────────────────────────────────────────
# Block factory
# ──────────────────────────────────────────────────────────────────────────────

_block_counter = 0


def make_block(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    block_type: BlockType = BlockType.TEXT,
    page_index: int = 0,
    block_id: str | None = None,
    detector_label: str = "text",
    confidence: float = 0.95,
    reading_order: int = -1,
) -> DocumentBlock:
    global _block_counter
    _block_counter += 1
    return DocumentBlock(
        block_id=block_id or f"b{_block_counter}",
        page_index=page_index,
        bbox=make_bbox(x1, y1, x2, y2),
        block_type=block_type,
        detector_label=detector_label,
        confidence=confidence,
        reading_order=reading_order,
    )


@pytest.fixture(autouse=True)
def reset_block_counter():
    """Reset block counter before each test for reproducible IDs."""
    global _block_counter
    _block_counter = 0
    yield


# ──────────────────────────────────────────────────────────────────────────────
# Dummy page image (blank 1000×1400 BGR)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_page():
    """800×1100 white BGR image simulating a single document page."""
    return np.ones((1100, 800, 3), dtype=np.uint8) * 255


@pytest.fixture
def blank_pages(blank_page):
    return [blank_page]


# ──────────────────────────────────────────────────────────────────────────────
# A minimal AgentState with one page
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_state(blank_pages):
    return {
        "input_path": "fake.pdf",
        "run_id": "test-run",
        "pages": blank_pages,
        "page_sizes": [{"page_index": 0, "width": 800, "height": 1100}],
        "blocks": [],
        "warnings": [],
        "status": "running",
    }
