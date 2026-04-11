"""Tests for types.py — BoundingBox, DocumentBlock, BlockType."""
from __future__ import annotations

import pytest

from document_agent.types import BlockType, BoundingBox, DocumentBlock
from conftest import make_block


class TestBoundingBox:
    def test_width_height(self):
        bb = BoundingBox(10, 20, 110, 70)
        assert bb.width == 100
        assert bb.height == 50

    def test_area(self):
        bb = BoundingBox(0, 0, 40, 50)
        assert bb.area == 2000

    def test_centroid(self):
        bb = BoundingBox(0, 0, 100, 200)
        assert bb.cx == 50.0
        assert bb.cy == 100.0

    def test_zero_size(self):
        bb = BoundingBox(5, 5, 5, 5)
        assert bb.width == 0
        assert bb.height == 0
        assert bb.area == 0

    def test_inverted_coords_clamp_to_zero(self):
        # x2 < x1 should give 0, not negative
        bb = BoundingBox(50, 50, 10, 10)
        assert bb.width == 0
        assert bb.height == 0
        assert bb.area == 0


class TestBlockType:
    def test_all_types_are_strings(self):
        for bt in BlockType:
            assert isinstance(bt.value, str)

    def test_expected_values(self):
        assert BlockType.TEXT.value == "text"
        assert BlockType.TABLE.value == "table"
        assert BlockType.MIXED.value == "mixed"
        assert BlockType.HEADER.value == "header"
        assert BlockType.FOOTER.value == "footer"
        assert BlockType.FORMULA.value == "formula"


class TestDocumentBlock:
    def test_defaults(self):
        b = make_block(0, 0, 100, 100)
        assert b.reading_order == -1
        assert b.parent_id is None
        assert b.child_ids == []
        assert b.relations == {}
        assert b.payload == {}
        assert b.line_group_id is None
        assert b.skip_specialist is False

    def test_as_dict_keys(self):
        b = make_block(10, 20, 110, 120, block_type=BlockType.CHART)
        d = b.as_dict()
        assert d["type"] == "chart"
        assert d["bbox"] == {"x1": 10, "y1": 20, "x2": 110, "y2": 120}
        assert "reading_order" in d
        assert "relations" in d
        assert "payload" in d
        assert "line_group_id" in d

    def test_as_dict_bbox_values(self):
        b = make_block(5, 10, 55, 60)
        d = b.as_dict()
        assert d["bbox"]["x1"] == 5
        assert d["bbox"]["y2"] == 60

    def test_child_ids_independence(self):
        b1 = make_block(0, 0, 50, 50)
        b2 = make_block(0, 0, 50, 50)
        b1.child_ids.append("x")
        assert "x" not in b2.child_ids

    def test_relations_independence(self):
        b1 = make_block(0, 0, 50, 50)
        b2 = make_block(0, 0, 50, 50)
        b1.relations["key"] = "val"
        assert "key" not in b2.relations
