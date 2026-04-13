"""
IO utilities for loading and saving document pages.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pymupdf


def _pixmap_to_bgr(pix: pymupdf.Pixmap) -> np.ndarray:
    """
    Convert a pymupdf.Pixmap to a BGR numpy array.
    """
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported number of channels: {pix.n}")


def load_document_pages(
    input_path: str,
    dpi: int = 200,
    max_pages: int = 0,
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """
    Load pages from a document file. Supports PDFs (rendered to images) and image files.
    Returns a list of page images as numpy arrays and their sizes.
    Each size dict contains: {"page_index": int, "width": int, "height": int}.

    max_pages: if > 0, only the first N pages are loaded (useful for benchmarking).
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input document not found: {path}")

    pages: List[np.ndarray] = []
    sizes: List[Dict[str, int]] = []

    if path.suffix.lower() == ".pdf":
        doc = pymupdf.open(path)
        try:
            zoom = dpi / 72.0 # 72 DPI is the default resolution of PDF points. Resolution scaled for pdf rendering.
            matrix = pymupdf.Matrix(zoom, zoom)
            for i, page in enumerate(doc):
                if max_pages > 0 and i >= max_pages:
                    break
                pix = page.get_pixmap(matrix=matrix, alpha=False) # Render page to image at specified DPI, without alpha channel (take a photo from the page)
                img = _pixmap_to_bgr(pix) # Convert to BGR format for OpenCV compatibility
                pages.append(img)   # Add to the list
                sizes.append({"page_index": i, "width": int(img.shape[1]), "height": int(img.shape[0])}) # Store page size info
        finally:
            doc.close()
    else:
        img = cv2.imread(str(path)) # Read image file directly (supports various formats like PNG, JPEG, TIFF, etc.)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        pages.append(img) # Add to the list
        sizes.append({"page_index": 0, "width": int(img.shape[1]), "height": int(img.shape[0])}) # Store page size info 

    return pages, sizes

