"""Tests for hierarchy.py — IoU, same-line, inline math stitching, caption linking."""
from __future__ import annotations

import pytest

from document_agent.hierarchy import (
    _iou,
    _link_captions,
    _overlap_ratio,
    _same_line,
    _stitch_inline_math,
    _x_adjacent,
    build_hierarchy,
)
from document_agent.types import BlockType
from conftest import make_block


# ──────────────────────────────────────────────────────────────────────────────
# _iou
# ──────────────────────────────────────────────────────────────────────────────

class TestIou:
    def test_identical_blocks(self):
        a = make_block(0, 0, 100, 100)
        b = make_block(0, 0, 100, 100)
        assert _iou(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = make_block(0, 0, 100, 100)
        b = make_block(200, 200, 300, 300)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = make_block(0, 0, 100, 100)   # area=10000
        b = make_block(50, 0, 150, 100)  # area=10000, overlap=50x100=5000
        # iou = 5000 / (10000+10000-5000) = 5000/15000 = 1/3
        assert _iou(a, b) == pytest.approx(1 / 3, rel=1e-3)

    def test_symmetry(self):
        a = make_block(0, 0, 80, 80)
        b = make_block(40, 40, 120, 120)
        assert _iou(a, b) == pytest.approx(_iou(b, a))


# ──────────────────────────────────────────────────────────────────────────────
# _overlap_ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestOverlapRatio:
    def test_child_fully_inside_parent(self):
        parent = make_block(0, 0, 200, 200)
        child  = make_block(50, 50, 150, 150)
        assert _overlap_ratio(child, parent) == pytest.approx(1.0)

    def test_no_overlap(self):
        parent = make_block(0, 0, 100, 100)
        child  = make_block(200, 200, 300, 300)
        assert _overlap_ratio(child, parent) == pytest.approx(0.0)

    def test_partial(self):
        parent = make_block(0, 0, 100, 100)
        child  = make_block(50, 0, 150, 100)   # half of child is inside parent
        assert _overlap_ratio(child, parent) == pytest.approx(0.5, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# _same_line
# ──────────────────────────────────────────────────────────────────────────────

class TestSameLine:
    def test_perfectly_aligned(self):
        a = make_block(0,  100, 200, 130)   # cy=115
        b = make_block(210, 100, 400, 130)  # cy=115
        assert _same_line(a, b) is True

    def test_clearly_different_rows(self):
        a = make_block(0, 100, 200, 130)    # cy=115
        b = make_block(0, 400, 200, 430)    # cy=415
        assert _same_line(a, b) is False

    def test_slightly_offset_still_same_line(self):
        # heights=30, tolerance=0.5 → max delta = 15
        a = make_block(0,   100, 200, 130)  # cy=115
        b = make_block(210, 108, 400, 138)  # cy=123 → delta=8 < 15
        assert _same_line(a, b) is True

    def test_zero_height_block(self):
        a = make_block(0, 100, 200, 100)    # height=0 → should not crash
        b = make_block(0, 100, 200, 130)
        assert _same_line(a, b) is False


# ──────────────────────────────────────────────────────────────────────────────
# _x_adjacent
# ──────────────────────────────────────────────────────────────────────────────

class TestXAdjacent:
    def test_touching_blocks(self):
        a = make_block(0,  100, 200, 130)
        b = make_block(200, 100, 400, 130)
        assert _x_adjacent(a, b) is True

    def test_small_gap(self):
        # gap=10, min_h=30, ratio=1.5 → max_gap=45 → adjacent
        a = make_block(0,  100, 200, 130)
        b = make_block(210, 100, 400, 130)
        assert _x_adjacent(a, b) is True

    def test_large_gap(self):
        # gap=200, min_h=30, ratio=1.5 → max_gap=45 → not adjacent
        a = make_block(0,  100, 200, 130)
        b = make_block(400, 100, 600, 130)
        assert _x_adjacent(a, b) is False


# ──────────────────────────────────────────────────────────────────────────────
# _stitch_inline_math
# ──────────────────────────────────────────────────────────────────────────────

class TestStitchInlineMath:
    def _make_inline_pair(self):
        """Text block [0,100,200,130] + formula block [205,100,280,130] on same line."""
        text    = make_block(0,   100, 200, 130, block_type=BlockType.TEXT)
        formula = make_block(205, 100, 280, 130, block_type=BlockType.FORMULA)
        return text, formula

    def test_creates_mixed_block(self):
        text, formula = self._make_inline_pair()
        result = _stitch_inline_math([text, formula])
        mixed_blocks = [b for b in result if b.block_type == BlockType.MIXED]
        assert len(mixed_blocks) == 1

    def test_mixed_block_covers_both(self):
        text, formula = self._make_inline_pair()
        result = _stitch_inline_math([text, formula])
        mixed = next(b for b in result if b.block_type == BlockType.MIXED)
        assert mixed.bbox.x1 == min(text.bbox.x1, formula.bbox.x1)
        assert mixed.bbox.x2 == max(text.bbox.x2, formula.bbox.x2)

    def test_children_marked_skip_specialist(self):
        text, formula = self._make_inline_pair()
        _stitch_inline_math([text, formula])
        assert text.skip_specialist is True
        assert formula.skip_specialist is True

    def test_children_get_line_group_id(self):
        text, formula = self._make_inline_pair()
        result = _stitch_inline_math([text, formula])
        mixed = next(b for b in result if b.block_type == BlockType.MIXED)
        assert text.line_group_id == mixed.block_id
        assert formula.line_group_id == mixed.block_id

    def test_no_mixed_when_different_rows(self):
        text    = make_block(0,   100, 200, 130, block_type=BlockType.TEXT)
        formula = make_block(0,   300, 200, 330, block_type=BlockType.FORMULA)
        result = _stitch_inline_math([text, formula])
        mixed_blocks = [b for b in result if b.block_type == BlockType.MIXED]
        assert len(mixed_blocks) == 0

    def test_no_mixed_when_large_x_gap(self):
        text    = make_block(0,   100, 200, 130, block_type=BlockType.TEXT)
        formula = make_block(500, 100, 650, 130, block_type=BlockType.FORMULA)
        result = _stitch_inline_math([text, formula])
        mixed_blocks = [b for b in result if b.block_type == BlockType.MIXED]
        assert len(mixed_blocks) == 0

    def test_no_formula_blocks_returns_unchanged(self):
        text = make_block(0, 100, 200, 130, block_type=BlockType.TEXT)
        result = _stitch_inline_math([text])
        assert len(result) == 1

    def test_multiple_formulas_on_same_line(self):
        """Two formulas adjacent to one text block → one MIXED block."""
        text  = make_block(0,   100, 200, 130, block_type=BlockType.TEXT)
        f1    = make_block(205, 100, 280, 130, block_type=BlockType.FORMULA)
        f2    = make_block(285, 100, 360, 130, block_type=BlockType.FORMULA)
        result = _stitch_inline_math([text, f1, f2])
        mixed_blocks = [b for b in result if b.block_type == BlockType.MIXED]
        # Both formulas partner with the same text block
        assert len(mixed_blocks) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# _link_captions
# ──────────────────────────────────────────────────────────────────────────────

class TestLinkCaptions:
    def test_caption_links_to_nearest_figure(self):
        figure  = make_block(50, 100, 400, 400, block_type=BlockType.FIGURE)
        far_img = make_block(50, 700, 400, 900, block_type=BlockType.IMAGE)
        caption = make_block(50, 420, 400, 450, block_type=BlockType.CAPTION)
        _link_captions([figure, far_img, caption])
        assert caption.relations.get("describes") == figure.block_id

    def test_figure_gets_caption_backlink(self):
        figure  = make_block(50, 100, 400, 400, block_type=BlockType.FIGURE)
        caption = make_block(50, 420, 400, 450, block_type=BlockType.CAPTION)
        _link_captions([figure, caption])
        assert caption.block_id in figure.relations.get("captions", [])

    def test_no_caption_no_change(self):
        figure = make_block(50, 100, 400, 400, block_type=BlockType.FIGURE)
        _link_captions([figure])
        assert figure.relations == {}

    def test_table_also_linked(self):
        table   = make_block(50, 100, 400, 400, block_type=BlockType.TABLE)
        caption = make_block(50, 420, 400, 450, block_type=BlockType.CAPTION)
        _link_captions([table, caption])
        assert caption.relations.get("describes") == table.block_id


# ──────────────────────────────────────────────────────────────────────────────
# build_hierarchy (integration)
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildHierarchy:
    def test_returns_all_original_blocks_plus_mixed(self):
        text    = make_block(0,   100, 200, 130, block_type=BlockType.TEXT)
        formula = make_block(205, 100, 280, 130, block_type=BlockType.FORMULA)
        result  = build_hierarchy([text, formula])
        # Original blocks + 1 MIXED
        assert len(result) == 3

    def test_containment_sets_parent(self):
        outer = make_block(0, 0, 400, 400, block_type=BlockType.FIGURE)
        inner = make_block(50, 50, 350, 350, block_type=BlockType.CAPTION)
        result = build_hierarchy([outer, inner])
        inner_result = next(b for b in result if b.block_id == inner.block_id)
        assert inner_result.parent_id == outer.block_id

    def test_no_parent_for_non_contained(self):
        a = make_block(0,   0, 100, 100)
        b = make_block(200, 0, 300, 100)
        result = build_hierarchy([a, b])
        for block in result:
            assert block.parent_id is None
