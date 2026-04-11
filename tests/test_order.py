"""Tests for order.py — column detection, heuristic ordering, LayoutReader fallback."""
from __future__ import annotations

import pytest

from document_agent.order import (
    _assign_column,
    _detect_column_boundaries,
    _heuristic_order,
    _is_full_width,
    _normalize_boxes,
    apply_reading_order,
)
from document_agent.types import BlockType
from conftest import make_block


PAGE_W = 800
PAGE_H = 1100


# ──────────────────────────────────────────────────────────────────────────────
# _is_full_width
# ──────────────────────────────────────────────────────────────────────────────

class TestIsFullWidth:
    def test_full_width_block(self):
        b = make_block(0, 0, 800, 50)          # 100 % of 800
        assert _is_full_width(b, PAGE_W) is True

    def test_half_width_block(self):
        b = make_block(0, 0, 400, 50)          # 50 %
        assert _is_full_width(b, PAGE_W) is False

    def test_threshold_boundary(self):
        b = make_block(0, 0, 520, 50)          # 65 % — exactly at threshold
        assert _is_full_width(b, PAGE_W) is True

    def test_just_below_threshold(self):
        b = make_block(0, 0, 519, 50)          # <65 %
        assert _is_full_width(b, PAGE_W) is False


# ──────────────────────────────────────────────────────────────────────────────
# _detect_column_boundaries
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectColumnBoundaries:
    def test_single_column_no_gap(self):
        # One block spanning the full width → no column separator
        blocks = [make_block(0, 0, PAGE_W, 100)]
        seps = _detect_column_boundaries(blocks, PAGE_W)
        assert seps == []

    def test_two_columns_clear_gap(self):
        # Left column [50,350], right column [450,750], gap at ~400
        left = make_block(50, 0, 350, 1000)
        right = make_block(450, 0, 750, 1000)
        seps = _detect_column_boundaries([left, right], PAGE_W)
        assert len(seps) == 1
        # Separator should sit somewhere in the gap [350, 450]
        assert 350 <= seps[0] <= 450

    def test_empty_blocks(self):
        assert _detect_column_boundaries([], PAGE_W) == []

    def test_no_gap_in_margin_excluded_zone(self):
        # Block covers 0-670; gap is 670-800.
        # right_margin = 0.80 * 800 = 640, so the gap starts at 670 > 640
        # → the gap is entirely outside the central zone and must NOT be detected.
        block = make_block(0, 0, 670, 1000)
        seps = _detect_column_boundaries([block], PAGE_W)
        assert seps == []


# ──────────────────────────────────────────────────────────────────────────────
# _assign_column
# ──────────────────────────────────────────────────────────────────────────────

class TestAssignColumn:
    def test_no_separators_always_col_0(self):
        b = make_block(400, 0, 500, 100)
        assert _assign_column(b, []) == 0

    def test_left_of_separator(self):
        b = make_block(100, 0, 300, 100)       # cx=200, separator at 400
        assert _assign_column(b, [400]) == 0

    def test_right_of_separator(self):
        b = make_block(450, 0, 700, 100)       # cx=575, separator at 400
        assert _assign_column(b, [400]) == 1

    def test_three_columns(self):
        b_left   = make_block(50,  0, 200, 100)   # cx=125
        b_mid    = make_block(300, 0, 450, 100)   # cx=375
        b_right  = make_block(600, 0, 750, 100)   # cx=675
        seps = [250, 525]
        assert _assign_column(b_left,  seps) == 0
        assert _assign_column(b_mid,   seps) == 1
        assert _assign_column(b_right, seps) == 2


# ──────────────────────────────────────────────────────────────────────────────
# _heuristic_order — two-column layout
# ──────────────────────────────────────────────────────────────────────────────

class TestHeuristicOrder:
    def _make_two_col_page(self):
        """
        Title spanning full width (y=0..50), then two columns:
          Left  col blocks: y=100, y=300, y=500
          Right col blocks: y=100, y=300, y=500
        """
        title  = make_block(0,   0,  800,  50, block_type=BlockType.TEXT, detector_label="title")
        l1     = make_block(50,  100, 350, 250, block_type=BlockType.TEXT)
        l2     = make_block(50,  300, 350, 450, block_type=BlockType.TEXT)
        l3     = make_block(50,  500, 350, 650, block_type=BlockType.TEXT)
        r1     = make_block(450, 100, 750, 250, block_type=BlockType.TEXT)
        r2     = make_block(450, 300, 750, 450, block_type=BlockType.TEXT)
        r3     = make_block(450, 500, 750, 650, block_type=BlockType.TEXT)
        return [title, l1, l2, l3, r1, r2, r3], title, [l1, l2, l3], [r1, r2, r3]

    def test_title_comes_first(self):
        blocks, title, left_col, right_col = self._make_two_col_page()
        order = _heuristic_order(blocks, PAGE_W, PAGE_H)
        title_rank = order.index(blocks.index(title))
        assert title_rank == 0

    def test_left_column_before_right(self):
        blocks, title, left_col, right_col = self._make_two_col_page()
        order = _heuristic_order(blocks, PAGE_W, PAGE_H)
        # All left col blocks should appear before all right col blocks
        left_ranks  = [order.index(blocks.index(b)) for b in left_col]
        right_ranks = [order.index(blocks.index(b)) for b in right_col]
        assert max(left_ranks) < min(right_ranks)

    def test_within_column_top_to_bottom(self):
        blocks, title, left_col, right_col = self._make_two_col_page()
        order = _heuristic_order(blocks, PAGE_W, PAGE_H)
        left_ranks = [order.index(blocks.index(b)) for b in left_col]
        assert left_ranks == sorted(left_ranks)

    def test_header_first_footer_last(self):
        header = make_block(0,   0,  800,  30, block_type=BlockType.HEADER)
        body   = make_block(50, 100, 750, 900, block_type=BlockType.TEXT)
        footer = make_block(0, 1050, 800, 1100, block_type=BlockType.FOOTER)
        blocks = [body, footer, header]
        order = _heuristic_order(blocks, PAGE_W, PAGE_H)
        assert order.index(blocks.index(header)) < order.index(blocks.index(body))
        assert order.index(blocks.index(body))   < order.index(blocks.index(footer))

    def test_single_column_top_to_bottom(self):
        b1 = make_block(100, 100, 700, 200)
        b2 = make_block(100, 250, 700, 350)
        b3 = make_block(100, 400, 700, 500)
        blocks = [b3, b1, b2]
        order = _heuristic_order(blocks, PAGE_W, PAGE_H)
        # b1 (y=100) first, b2 (y=250) second, b3 (y=400) last
        assert order == [1, 2, 0]

    def test_returns_valid_permutation(self):
        blocks = [make_block(i * 100, 0, i * 100 + 80, 50) for i in range(5)]
        order = _heuristic_order(blocks, PAGE_W, PAGE_H)
        assert sorted(order) == list(range(len(blocks)))


# ──────────────────────────────────────────────────────────────────────────────
# _normalize_boxes
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeBoxes:
    def test_full_page_block(self):
        b = make_block(0, 0, PAGE_W, PAGE_H)
        boxes = _normalize_boxes([b], PAGE_W, PAGE_H)
        assert boxes == [[0, 0, 1000, 1000]]

    def test_clamps_to_1000(self):
        b = make_block(0, 0, PAGE_W + 100, PAGE_H + 100)
        boxes = _normalize_boxes([b], PAGE_W, PAGE_H)
        assert boxes[0][2] == 1000
        assert boxes[0][3] == 1000

    def test_proportional(self):
        b = make_block(0, 0, PAGE_W // 2, PAGE_H // 2)
        boxes = _normalize_boxes([b], PAGE_W, PAGE_H)
        assert boxes == [[0, 0, 500, 500]]


# ──────────────────────────────────────────────────────────────────────────────
# apply_reading_order (integration — LayoutReader mocked to be unavailable)
# ──────────────────────────────────────────────────────────────────────────────

class TestApplyReadingOrder:
    def test_sets_reading_order_on_all_blocks(self, monkeypatch):
        monkeypatch.setattr("document_agent.order._layoutreader_predict", lambda *a, **k: None)
        blocks = [make_block(50, i * 100, 750, i * 100 + 80) for i in range(4)]
        page_sizes = [{"page_index": 0, "width": PAGE_W, "height": PAGE_H}]
        result = apply_reading_order(blocks, page_sizes)
        orders = [b.reading_order for b in result]
        assert all(o >= 0 for o in orders)
        assert sorted(orders) == list(range(len(blocks)))

    def test_multipage_reading_order_is_per_page(self, monkeypatch):
        monkeypatch.setattr("document_agent.order._layoutreader_predict", lambda *a, **k: None)
        p0 = [make_block(50, i * 100, 750, i * 100 + 80, page_index=0) for i in range(3)]
        p1 = [make_block(50, i * 100, 750, i * 100 + 80, page_index=1) for i in range(2)]
        page_sizes = [
            {"page_index": 0, "width": PAGE_W, "height": PAGE_H},
            {"page_index": 1, "width": PAGE_W, "height": PAGE_H},
        ]
        result = apply_reading_order(p0 + p1, page_sizes)
        p0_orders = sorted(b.reading_order for b in result if b.page_index == 0)
        p1_orders = sorted(b.reading_order for b in result if b.page_index == 1)
        assert p0_orders == [0, 1, 2]
        assert p1_orders == [0, 1]
