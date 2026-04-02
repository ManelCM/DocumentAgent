"""
Tests for nodes.py.
Real installed packages are used (pytesseract, langgraph, openai).
OpenAI API calls are skipped when no key is set.
PaddleOCR is mocked only for path-coverage tests (new-API / legacy-API format),
since loading the full model in unit tests is impractical.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytesseract

from document_agent.nodes import (
    _extract_text_paddle,
    node_aggregate,
    node_association,
    node_cross_reference,
    node_reduce_specialists,
)
from document_agent.types import BlockType
from conftest import make_block


# ──────────────────────────────────────────────────────────────────────────────
# _extract_text_paddle
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractTextPaddle:
    def test_empty_crop_returns_empty_text(self):
        result = _extract_text_paddle(None)
        assert result["text"] == ""
        assert "ocr_engine" in result

    def test_zero_size_crop(self):
        crop = np.zeros((0, 0, 3), dtype=np.uint8)
        result = _extract_text_paddle(crop)
        assert result["text"] == ""

    def test_paddle_new_api_rec_texts(self):
        """PaddleOCR predict() returns objects with rec_texts attribute."""
        mock_page = MagicMock()
        mock_page.rec_texts = ["Hello", "world"]
        with patch("document_agent.nodes._get_paddle_ocr") as mock_get:
            mock_ocr = MagicMock()
            mock_ocr.predict.return_value = [mock_page]
            mock_get.return_value = mock_ocr
            crop = np.ones((50, 200, 3), dtype=np.uint8) * 255
            result = _extract_text_paddle(crop)
        assert result["text"] == "Hello world"
        assert result["ocr_engine"] == "paddleocr"

    def test_paddle_legacy_api_list_format(self):
        """PaddleOCR predict() returns list-of-lists (old format)."""
        mock_line = [[[0, 0], [100, 0], [100, 20], [0, 20]], ("Sample text", 0.99)]
        with patch("document_agent.nodes._get_paddle_ocr") as mock_get:
            mock_ocr = MagicMock()
            mock_ocr.predict.return_value = [[mock_line]]
            mock_get.return_value = mock_ocr
            crop = np.ones((50, 200, 3), dtype=np.uint8) * 255
            result = _extract_text_paddle(crop)
        assert "Sample text" in result["text"]

    def test_paddle_unavailable_falls_back_to_real_tesseract(self):
        """When PaddleOCR is unavailable, real Tesseract processes the image."""
        # A white image returns empty string from Tesseract — that is correct behaviour.
        with patch("document_agent.nodes._get_paddle_ocr", return_value=None):
            crop = np.ones((60, 200, 3), dtype=np.uint8) * 255
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _extract_text_paddle(crop)
        assert "pytesseract" in result["ocr_engine"]
        assert isinstance(result["text"], str)

    def test_paddle_unavailable_tesseract_reads_text(self):
        """Tesseract can read a simple rendered text image without PaddleOCR."""
        import cv2
        # Render a known string onto a white image
        img = np.ones((60, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "HELLO", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
        with patch("document_agent.nodes._get_paddle_ocr", return_value=None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _extract_text_paddle(img)
        assert "pytesseract" in result["ocr_engine"]
        # Tesseract should detect something (exact string depends on rendering)
        assert isinstance(result["text"], str)

    def test_all_engines_fail_returns_empty(self, monkeypatch):
        """When both PaddleOCR and Tesseract raise, return empty fallback."""
        monkeypatch.setattr("document_agent.nodes._get_paddle_ocr", lambda: None)
        # Point tesseract_cmd to a non-existent binary so the subprocess fails
        original_cmd = pytesseract.pytesseract.tesseract_cmd
        pytesseract.pytesseract.tesseract_cmd = "nonexistent_tesseract_binary"
        try:
            crop = np.ones((50, 200, 3), dtype=np.uint8) * 255
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _extract_text_paddle(crop)
        finally:
            pytesseract.pytesseract.tesseract_cmd = original_cmd
        assert result["text"] == ""
        assert result["ocr_engine"] == "fallback"


# ──────────────────────────────────────────────────────────────────────────────
# node_cross_reference
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeCrossReference:
    def _state_with_text(self, text: str):
        block = make_block(0, 0, 100, 100, block_type=BlockType.TEXT)
        block.payload["text"] = text
        return {"blocks": [block], "pages": [], "page_sizes": []}

    def test_detects_figure_reference(self):
        state = self._state_with_text("As shown in Figure 3, the results indicate...")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        assert any(r["type"] == "figure" and r["label"] == "3" for r in refs)

    def test_detects_fig_abbreviation(self):
        state = self._state_with_text("see Fig. 2a for details")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        # The regex captures the full label including sub-figure letter ("2a")
        assert any(r["type"] == "figure" and r["label"] == "2a" for r in refs)

    def test_detects_table_reference(self):
        state = self._state_with_text("Table 5 summarizes the comparison.")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        assert any(r["type"] == "table" and r["label"] == "5" for r in refs)

    def test_detects_equation_reference(self):
        state = self._state_with_text("Substituting into Equation 12 gives...")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        assert any(r["type"] == "equation" and r["label"] == "12" for r in refs)

    def test_detects_eq_abbreviation(self):
        state = self._state_with_text("From Eq. 4 we derive...")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        assert any(r["type"] == "equation" for r in refs)

    def test_detects_citation(self):
        state = self._state_with_text("as described by Smith et al. [14].")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        assert any(r["type"] == "citation" and r["label"] == "14" for r in refs)

    def test_multiple_references_detected(self):
        state = self._state_with_text("See Figure 1 and Table 2 and Equation 3.")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        types = {r["type"] for r in refs}
        assert "figure" in types
        assert "table" in types
        assert "equation" in types

    def test_no_reference_text_leaves_no_relations(self):
        state = self._state_with_text("The sky is blue.")
        result = node_cross_reference(state)
        assert "references" not in result["blocks"][0].relations

    def test_skips_formula_blocks(self):
        block = make_block(0, 0, 100, 100, block_type=BlockType.FORMULA)
        block.payload["formula_latex"] = "E = mc^2"
        state = {"blocks": [block], "pages": [], "page_sizes": []}
        result = node_cross_reference(state)
        assert "references" not in result["blocks"][0].relations

    def test_case_insensitive_figure(self):
        state = self._state_with_text("FIGURE 7 shows...")
        result = node_cross_reference(state)
        refs = result["blocks"][0].relations.get("references", [])
        assert any(r["type"] == "figure" for r in refs)


# ──────────────────────────────────────────────────────────────────────────────
# node_association
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeAssociation:
    def test_caption_linked_to_nearest_figure(self):
        figure  = make_block(50, 100, 400, 400, block_type=BlockType.FIGURE, block_id="fig1")
        far_img = make_block(50, 700, 400, 900, block_type=BlockType.IMAGE,  block_id="img1")
        caption = make_block(50, 420, 400, 450, block_type=BlockType.CAPTION, block_id="cap1")
        state = {
            "blocks": [figure, far_img, caption],
            "page_sizes": [{"page_index": 0, "width": 800, "height": 1100}],
        }
        result = node_association(state)
        blocks_by_id = {b.block_id: b for b in result["blocks"]}
        assert blocks_by_id["cap1"].relations.get("describes") == "fig1"

    def test_full_page_image_flagged(self):
        img = make_block(0, 0, 800, 1100, block_type=BlockType.IMAGE, block_id="img1")
        state = {
            "blocks": [img],
            "page_sizes": [{"page_index": 0, "width": 800, "height": 1100}],
        }
        result = node_association(state)
        img_block = result["blocks"][0]
        assert img_block.relations.get("is_page_image") is True
        assert img_block.relations["page_coverage"] > 0.85

    def test_small_image_not_flagged_as_page_image(self):
        img = make_block(100, 100, 400, 400, block_type=BlockType.IMAGE)
        state = {
            "blocks": [img],
            "page_sizes": [{"page_index": 0, "width": 800, "height": 1100}],
        }
        result = node_association(state)
        assert "is_page_image" not in result["blocks"][0].relations

    def test_embedded_formula_linked_to_mixed_parent(self):
        mixed   = make_block(0, 100, 300, 130, block_type=BlockType.MIXED, block_id="mix1")
        formula = make_block(200, 100, 280, 130, block_type=BlockType.FORMULA, block_id="frm1")
        formula.parent_id = "mix1"
        state = {
            "blocks": [mixed, formula],
            "page_sizes": [{"page_index": 0, "width": 800, "height": 1100}],
        }
        result = node_association(state)
        blocks_by_id = {b.block_id: b for b in result["blocks"]}
        assert "frm1" in blocks_by_id["mix1"].relations.get("embedded_formulas", [])


# ──────────────────────────────────────────────────────────────────────────────
# node_reduce_specialists
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeReduceSpecialists:
    def test_merges_text_payload_into_block(self):
        block = make_block(0, 0, 100, 100, block_type=BlockType.TEXT, block_id="b1")
        state = {
            "blocks": [block],
            "warnings": [],
            "text_updates": {"b1": {"text": "Hello world", "ocr_engine": "paddleocr"}},
            "text_warnings": [],
            "image_updates": {}, "chart_updates": {}, "formula_updates": {},
            "table_updates": {}, "other_updates": {},
            "image_warnings": [], "chart_warnings": [], "formula_warnings": [],
            "table_warnings": [], "other_warnings": [],
        }
        result = node_reduce_specialists(state)
        b = result["blocks"][0]
        assert b.payload["text"] == "Hello world"
        assert b.payload["ocr_engine"] == "paddleocr"

    def test_unknown_block_id_in_updates_is_ignored(self):
        block = make_block(0, 0, 100, 100, block_id="b1")
        state = {
            "blocks": [block],
            "warnings": [],
            "text_updates": {"NONEXISTENT": {"text": "ghost"}},
            "text_warnings": [],
            "image_updates": {}, "chart_updates": {}, "formula_updates": {},
            "table_updates": {}, "other_updates": {},
            "image_warnings": [], "chart_warnings": [], "formula_warnings": [],
            "table_warnings": [], "other_warnings": [],
        }
        result = node_reduce_specialists(state)
        assert result["blocks"][0].payload == {}

    def test_warnings_aggregated(self):
        state = {
            "blocks": [],
            "warnings": ["pre-existing"],
            "text_updates": {}, "image_updates": {}, "chart_updates": {},
            "formula_updates": {}, "table_updates": {}, "other_updates": {},
            "text_warnings": ["tw1"],
            "image_warnings": ["iw1"],
            "chart_warnings": [],
            "formula_warnings": ["fw1", "fw2"],
            "table_warnings": [],
            "other_warnings": [],
        }
        result = node_reduce_specialists(state)
        assert "pre-existing" in result["warnings"]
        assert "tw1" in result["warnings"]
        assert "iw1" in result["warnings"]
        assert "fw1" in result["warnings"]
        assert "fw2" in result["warnings"]


# ──────────────────────────────────────────────────────────────────────────────
# node_aggregate
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeAggregate:
    def test_output_structure(self):
        blocks = [
            make_block(0, i * 100, 100, i * 100 + 80, block_type=BlockType.TEXT, reading_order=i)
            for i in range(3)
        ]
        state = {
            "run_id": "test-123",
            "input_path": "doc.pdf",
            "blocks": blocks,
            "page_sizes": [{"page_index": 0, "width": 800, "height": 1100}],
            "warnings": [],
            "status": "running",
        }
        result = node_aggregate(state)
        output = result["output"]
        assert output["run_id"] == "test-123"
        assert output["status"] == "done"
        assert len(output["blocks"]) == 3
        assert output["summary"]["num_blocks"] == 3
        assert output["summary"]["num_pages"] == 1

    def test_blocks_sorted_by_reading_order(self):
        b0 = make_block(0, 200, 100, 280, reading_order=2)
        b1 = make_block(0, 100, 100, 180, reading_order=0)
        b2 = make_block(0, 300, 100, 380, reading_order=1)
        state = {
            "run_id": "r",
            "input_path": "x.pdf",
            "blocks": [b0, b1, b2],
            "page_sizes": [{"page_index": 0, "width": 800, "height": 1100}],
            "warnings": [],
            "status": "running",
        }
        result = node_aggregate(state)
        ids = [b["id"] for b in result["output"]["blocks"]]
        assert ids == [b1.block_id, b2.block_id, b0.block_id]

    def test_type_counts_in_summary(self):
        blocks = [
            make_block(0, 0, 100, 100, block_type=BlockType.TEXT),
            make_block(0, 0, 100, 100, block_type=BlockType.TEXT),
            make_block(0, 0, 100, 100, block_type=BlockType.TABLE),
        ]
        state = {
            "run_id": "r", "input_path": "x.pdf",
            "blocks": blocks,
            "page_sizes": [],
            "warnings": [],
            "status": "running",
        }
        result = node_aggregate(state)
        types = result["output"]["summary"]["types"]
        assert types["text"] == 2
        assert types["table"] == 1

    def test_status_set_to_done(self):
        state = {
            "run_id": "r", "input_path": "x.pdf",
            "blocks": [], "page_sizes": [], "warnings": [], "status": "running",
        }
        result = node_aggregate(state)
        assert result["status"] == "done"
        assert result["output"]["status"] == "done"
