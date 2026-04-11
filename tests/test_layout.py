"""Tests for layout.py — label mapping."""
from __future__ import annotations

import pytest

from document_agent.layout import map_detector_label
from document_agent.types import BlockType


class TestMapDetectorLabel:
    # ── Text family ──────────────────────────────────────────────────────────
    @pytest.mark.parametrize("label", [
        "text", "paragraph", "list", "reference", "abstract", "section",
        # PP-DocLayout_plus-L specific
        "reference_content", "doc_title", "aside_text",
    ])
    def test_text_labels(self, label):
        assert map_detector_label(label) == BlockType.TEXT

    @pytest.mark.parametrize("label", [
        "title", "heading", "headline", "section_title", "sub_title",
        "paragraph_title",  # PP-DocLayout_plus-L section heading
    ])
    def test_title_labels_map_to_text(self, label):
        # Titles are kept as TEXT (distinguished by detector_label at reading order time)
        assert map_detector_label(label) == BlockType.TEXT

    @pytest.mark.parametrize("label", ["formula_number"])
    def test_formula_number_maps_to_text(self, label):
        # Equation labels like "(1)" are tiny text, not full formulas
        assert map_detector_label(label) == BlockType.TEXT

    # ── Formula family ───────────────────────────────────────────────────────
    @pytest.mark.parametrize("label", ["formula", "equation", "inline_formula", "display_formula"])
    def test_formula_labels(self, label):
        assert map_detector_label(label) == BlockType.FORMULA

    # ── Table ────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("label", ["table", "table_caption"])
    def test_table_labels(self, label):
        assert map_detector_label(label) == BlockType.TABLE

    # ── Image / chart / figure ───────────────────────────────────────────────
    @pytest.mark.parametrize("label", ["image", "photo"])
    def test_image_labels(self, label):
        assert map_detector_label(label) == BlockType.IMAGE

    @pytest.mark.parametrize("label", ["chart", "graph", "plot"])
    def test_chart_labels(self, label):
        assert map_detector_label(label) == BlockType.CHART

    @pytest.mark.parametrize("label", ["figure", "subfigure"])
    def test_figure_labels(self, label):
        assert map_detector_label(label) == BlockType.FIGURE

    # ── Caption ──────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("label", [
        "caption", "figure_caption", "table_caption_text",
        "figure_title",   # PP-DocLayout_plus-L figure caption label
    ])
    def test_caption_labels(self, label):
        assert map_detector_label(label) == BlockType.CAPTION

    # ── Header / Footer ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("label", ["header", "page_header", "running_head", "running_title"])
    def test_header_labels(self, label):
        assert map_detector_label(label) == BlockType.HEADER

    @pytest.mark.parametrize("label", [
        "footer", "page_footer", "footnote", "page_number",
        "number",  # PP-DocLayout_plus-L standalone page number
    ])
    def test_footer_labels(self, label):
        assert map_detector_label(label) == BlockType.FOOTER

    # ── Unknown / Other ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("label", ["unknown", "garbage", "", "  "])
    def test_unknown_labels(self, label):
        assert map_detector_label(label) == BlockType.OTHER

    def test_case_insensitive(self):
        assert map_detector_label("TEXT") == BlockType.TEXT
        assert map_detector_label("Table") == BlockType.TABLE
        assert map_detector_label("FORMULA") == BlockType.FORMULA

    def test_none_like_input(self):
        assert map_detector_label("") == BlockType.OTHER
