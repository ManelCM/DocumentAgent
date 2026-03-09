from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pymupdf


def _pixmap_to_bgr(pix: pymupdf.Pixmap) -> np.ndarray:
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_document_pages(input_path: str, dpi: int = 200) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input document not found: {path}")

    pages: List[np.ndarray] = []
    sizes: List[Dict[str, int]] = []

    if path.suffix.lower() == ".pdf":
        doc = pymupdf.open(path)
        try:
            zoom = dpi / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)
            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = _pixmap_to_bgr(pix)
                pages.append(img)
                sizes.append({"page_index": i, "width": int(img.shape[1]), "height": int(img.shape[0])})
        finally:
            doc.close()
    else:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        pages.append(img)
        sizes.append({"page_index": 0, "width": int(img.shape[1]), "height": int(img.shape[0])})

    return pages, sizes

