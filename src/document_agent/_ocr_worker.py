"""
Subprocess worker for PaddleOCR text recognition.

Usage: python _ocr_worker.py <crops_pkl_path>

Reads a pickled list of {"block_id": str, "crop": ndarray} dicts,
runs PaddleOCR PP-OCRv5 on each crop, and writes a JSON result to
stdout: {"results": [{"block_id": str, "text": str, "confidence": float}, ...]}

Running OCR in a subprocess keeps the PaddleOCR PP-OCRv5 DLLs isolated
from the LayoutDetection DLLs that are already loaded in the main process
(they share PaddlePaddle's oneDNN/TBB backend and crash on Windows when
both initialise in the same process).
"""
# ── Path fix ──────────────────────────────────────────────────────────────────
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
while _here in _sys.path:
    _sys.path.remove(_here)
# ─────────────────────────────────────────────────────────────────────────────

import json
import pickle
import sys


def _run_ocr(crop):
    """Run PP-OCRv5 on a single crop. Returns (text, avg_confidence)."""
    from paddleocr import PaddleOCR
    # Lazy singleton — one per worker process
    global _OCR
    if "_OCR" not in globals():
        # PaddleOCR v3 removed show_log; use_angle_cls deprecated → use_textline_orientation
        try:
            _OCR = PaddleOCR(use_textline_orientation=False, lang="en")
        except TypeError:
            _OCR = PaddleOCR(use_angle_cls=False, lang="en")  # v2 fallback
    try:
        result = list(_OCR.predict(crop))  # v3 API
    except AttributeError:
        result = _OCR.ocr(crop, cls=False)  # v2 fallback
    lines, confs = [], []
    if result:
        for item in result:
            # v3: OCRResult objects with rec_text / rec_score attributes
            if hasattr(item, "rec_text"):
                lines.append(str(item.rec_text))
                confs.append(float(getattr(item, "rec_score", 1.0)))
            elif hasattr(item, "text"):
                lines.append(str(item.text))
                confs.append(float(getattr(item, "score", 1.0)))
            elif isinstance(item, (list, tuple)):
                # v2: [[box], [text, conf]] or page wrapper
                page_items = item if isinstance(item[0], (list, tuple)) else [item]
                for line in page_items:
                    if line and len(line) >= 2 and line[1]:
                        try:
                            txt, conf = line[1][0], line[1][1]
                            lines.append(str(txt))
                            confs.append(float(conf))
                        except (IndexError, TypeError):
                            pass
    text = " ".join(lines).strip()
    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
    return text, avg_conf


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing crops pkl path"}))
        sys.exit(1)

    pkl_path = sys.argv[1]
    with open(pkl_path, "rb") as f:
        items = pickle.load(f)   # list of {"block_id": str, "crop": ndarray}

    results = []
    for item in items:
        bid = item["block_id"]
        crop = item["crop"]
        try:
            text, conf = _run_ocr(crop)
        except Exception as exc:
            text, conf = "", 0.0
        results.append({"block_id": bid, "text": text, "confidence": conf,
                         "ocr_engine": "paddleocr"})

    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
