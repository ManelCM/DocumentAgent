"""
Subprocess worker for layout detection.

Run as: python _layout_worker.py <pages_pkl_path>

Reads a pickled list of numpy image arrays, runs LayoutDetection,
writes a JSON block list to stdout.

This keeps paddlex LayoutDetection DLLs isolated from PaddleOCR OCR
DLLs so they never share the same process (they segfault on Windows
when both are loaded together).
"""
# ── Path fix ──────────────────────────────────────────────────────────────────
# When Python runs a script directly it inserts the script's directory as
# sys.path[0].  document_agent/types.py would shadow the stdlib `types` module,
# breaking json, enum and many other stdlib imports.  Remove it immediately.
import os as _os, sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
while _here in _sys.path:
    _sys.path.remove(_here)
# ─────────────────────────────────────────────────────────────────────────────

import json
import pickle
import sys


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing pages pkl path"}))
        sys.exit(1)

    pkl_path = sys.argv[1]
    with open(pkl_path, "rb") as f:
        pages = pickle.load(f)

    try:
        from paddleocr import LayoutDetection
        detector = LayoutDetection()
    except Exception as exc:
        print(json.dumps({"error": f"LayoutDetection unavailable: {exc}"}))
        sys.exit(0)

    blocks = []
    block_num = 0
    for page_idx, img in enumerate(pages):
        page_h, page_w = img.shape[:2]
        try:
            result = detector.predict(img)
            page_boxes = result[0].get("boxes", []) if result else []
        except Exception:
            page_boxes = []

        if not page_boxes:
            block_num += 1
            blocks.append({
                "block_id": f"b{block_num}",
                "page_index": page_idx,
                "bbox": {"x1": 0, "y1": 0, "x2": page_w, "y2": page_h},
                "label": "page_empty_fallback",
                "score": 1.0,
            })
            continue

        for item in page_boxes:
            x1, y1, x2, y2 = [int(v) for v in item["coordinate"]]
            block_num += 1
            blocks.append({
                "block_id": f"b{block_num}",
                "page_index": page_idx,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "label": str(item.get("label", "other")),
                "score": float(item.get("score", 0.0)),
            })

    print(json.dumps({"blocks": blocks}))


if __name__ == "__main__":
    main()
