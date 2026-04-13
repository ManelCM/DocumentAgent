"""
OCR Engine Comparison Benchmark
================================
Runs multiple OCR engines on the same documents and compares:
  - CER / WER  (vs native PDF text layer as ground truth)
  - Speed      (seconds per page)
  - Cost       (estimated $/page where applicable)

Engines compared
----------------
  Tier 1 — Raw OCR only (no structure):
    - Tesseract    (via pytesseract)
    - EasyOCR      (via easyocr)
    - PaddleOCR    (via paddleocr)

  Tier 2 — Advanced / LLM-augmented:
    - Marker       (via marker_single)
    - Docling      (via docling)
    - DocumentAgent (existing output.json files — skip re-running if present)

Usage
-----
    python scripts/compare_engines.py [--docs-dir data/test_docs] [--output-dir output/engine_compare]
                                      [--max-pages 5] [--engines all] [--skip-docagent]

Ground truth
------------
Uses pymupdf to extract the native text layer from born-digital PDFs.
Scanned-only PDFs will have nearly empty text layers and are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("compare_engines")

# ── Ground-truth extraction ─────────────────────────────────────────────────────

def extract_gt_text(pdf_path: Path, max_pages: int = 0) -> Tuple[str, int]:
    """Extract native text layer using pymupdf as ground truth.
    Returns (text, num_pages_used).
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        log.error("pymupdf not installed: pip install pymupdf")
        return "", 0

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    pages = list(range(min(max_pages, total) if max_pages > 0 else total))
    texts = []
    for i in pages:
        texts.append(doc[i].get_text("text"))
    doc.close()
    return "\n".join(texts), len(pages)


def _is_born_digital(text: str, n_pages: int) -> bool:
    """Heuristic: avg chars/page > 200 → likely born-digital."""
    if n_pages == 0:
        return False
    return (len(text.strip()) / n_pages) > 200


# ── Text normalization for fair CER/WER ───────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip non-printable."""
    text = text.lower()
    text = re.sub(r"[^\x20-\x7e\n]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(hypothesis: str, reference: str) -> Dict[str, float]:
    """Return CER and WER using jiwer."""
    try:
        import jiwer
    except ImportError:
        return {"cer": -1.0, "wer": -1.0}

    ref = _normalize(reference)
    hyp = _normalize(hypothesis)

    if not ref.strip():
        return {"cer": -1.0, "wer": -1.0}

    cer = jiwer.cer(ref, hyp)
    wer = jiwer.wer(ref, hyp)
    return {"cer": round(float(cer), 4), "wer": round(float(wer), 4)}


# ── Render PDF pages to images ────────────────────────────────────────────────

def render_pages(pdf_path: Path, max_pages: int, dpi: int = 200) -> List[Any]:
    """Render first max_pages pages to PIL Images via pymupdf."""
    import fitz
    from PIL import Image
    import io

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    n = min(max_pages, total) if max_pages > 0 else total
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i in range(n):
        pix = doc[i].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def images_to_numpy(images: List[Any]):
    """Convert PIL images to numpy arrays."""
    import numpy as np
    return [np.array(img) for img in images]


# ── Engine runners ────────────────────────────────────────────────────────────

def run_tesseract(pdf_path: Path, max_pages: int) -> Tuple[str, float]:
    """Run Tesseract via pytesseract on rendered pages."""
    try:
        import pytesseract
    except ImportError:
        return "[tesseract not installed]", 0.0

    t0 = time.perf_counter()
    images = render_pages(pdf_path, max_pages)
    texts = []
    for img in images:
        texts.append(pytesseract.image_to_string(img, lang="eng"))
    elapsed = time.perf_counter() - t0
    return "\n".join(texts), elapsed


def run_easyocr(pdf_path: Path, max_pages: int) -> Tuple[str, float]:
    """Run EasyOCR on rendered pages."""
    try:
        import easyocr
    except ImportError:
        return "[easyocr not installed]", 0.0

    t0 = time.perf_counter()
    images = images_to_numpy(render_pages(pdf_path, max_pages))
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    texts = []
    for arr in images:
        results = reader.readtext(arr, detail=0)
        texts.append(" ".join(results))
    elapsed = time.perf_counter() - t0
    return "\n".join(texts), elapsed


def _extract_paddle_text(result) -> str:
    """Extract plain text from PaddleOCR result (handles v2 and v3 formats)."""
    if result is None:
        return ""
    lines = []
    try:
        # v3 returns a generator/list of OCRResult objects with .rec_text attribute
        for item in result:
            if hasattr(item, "rec_text"):
                lines.append(str(item.rec_text))
            elif hasattr(item, "text"):
                lines.append(str(item.text))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # v2 format: [[box], [text, conf]]
                text_part = item[1]
                if isinstance(text_part, (list, tuple)):
                    lines.append(str(text_part[0]))
                elif isinstance(text_part, str):
                    lines.append(text_part)
            elif isinstance(item, (list, tuple)):
                # v3 nested: list of pages
                for sub in item or []:
                    if sub is None:
                        continue
                    if hasattr(sub, "rec_text"):
                        lines.append(str(sub.rec_text))
                    elif isinstance(sub, (list, tuple)) and len(sub) >= 2:
                        tp = sub[1]
                        if isinstance(tp, (list, tuple)):
                            lines.append(str(tp[0]))
    except Exception:
        pass
    return " ".join(lines)


def run_paddleocr(pdf_path: Path, max_pages: int) -> Tuple[str, float]:
    """Run PaddleOCR on rendered pages (text only, no layout)."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return "[paddleocr not installed]", 0.0

    t0 = time.perf_counter()
    images = images_to_numpy(render_pages(pdf_path, max_pages))
    # PaddleOCR v3: use_angle_cls/show_log removed
    try:
        ocr = PaddleOCR(use_textline_orientation=False)
    except TypeError:
        ocr = PaddleOCR(use_angle_cls=False, lang="en")  # v2 fallback
    texts = []
    for arr in images:
        try:
            result = ocr.predict(arr)
        except AttributeError:
            result = ocr.ocr(arr, cls=False)
        texts.append(_extract_paddle_text(result))
    elapsed = time.perf_counter() - t0
    return "\n".join(texts), elapsed


def run_marker(pdf_path: Path, max_pages: int, output_dir: Path) -> Tuple[str, float]:
    """Run marker-pdf via Python API and return stripped markdown text."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
    except ImportError:
        return "[marker not installed]", 0.0

    t0 = time.perf_counter()
    try:
        config = {"output_format": "markdown"}
        if max_pages > 0:
            config["max_pages"] = max_pages

        config_parser = ConfigParser(config)
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=config_parser.generate_config_dict(),
        )
        rendered = converter(str(pdf_path))
        md = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)
        # Strip markdown formatting for plain text comparison
        text = re.sub(r"#{1,6}\s+", "", md)
        text = re.sub(r"\*\*|__|\*|_", "", text)
        text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
        text = re.sub(r"\$[^$]+\$", " ", text)
    except Exception as exc:
        return f"[marker error: {exc}]", 0.0
    elapsed = time.perf_counter() - t0
    return text, elapsed


def run_docling(pdf_path: Path, max_pages: int) -> Tuple[str, float]:
    """Run Docling converter and extract plain text."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return "[docling not installed]", 0.0

    t0 = time.perf_counter()
    try:
        converter = DocumentConverter()
        kwargs: Dict = {"raises_on_error": False}
        if max_pages > 0:
            kwargs["max_num_pages"] = max_pages
        result = converter.convert(str(pdf_path), **kwargs)
        # Export to plain text (no markdown formatting to strip)
        if hasattr(result.document, "export_to_text"):
            text = result.document.export_to_text()
        else:
            md = result.document.export_to_markdown()
            text = re.sub(r"#{1,6}\s+", "", md)
            text = re.sub(r"\*\*|__|\*|_", "", text)
            text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
            text = re.sub(r"\$[^$]+\$", " ", text)
    except Exception as exc:
        return f"[docling error: {exc}]", 0.0
    elapsed = time.perf_counter() - t0
    return text, elapsed


def _docagent_raw_text(data: Dict) -> str:
    """Extract raw concatenated block texts from DocumentAgent output.

    More comparable to other raw OCR engines: strips [PAGE N] markers,
    page-flow reordering, and structural additions. Just raw payload texts
    in reading order.
    """
    blocks = data.get("blocks", [])
    skip_types = {"header", "footer", "other"}
    ordered = sorted(
        blocks,
        key=lambda b: (b.get("page_index", 0), b.get("reading_order", 999_999)),
    )
    texts = []
    for b in ordered:
        if b.get("type") in skip_types:
            continue
        p = b.get("payload", {})
        text = (p.get("text") or p.get("full_text") or p.get("plain_text") or "").strip()
        if text and len(text) > 2:
            texts.append(text)
    return " ".join(texts)


def run_documentagent(
    pdf_path: Path,
    max_pages: int,
    output_dir: Path,
    skip_if_exists: bool = True,
) -> Tuple[str, float]:
    """Run DocumentAgent pipeline and return raw extracted block text.

    Uses raw payload text from blocks (not the structured full_text field)
    for a fairer CER/WER comparison against other OCR engines.
    """
    stem = pdf_path.stem
    job_dir = output_dir / "documentagent" / stem
    job_dir.mkdir(parents=True, exist_ok=True)
    output_json = job_dir / "output.json"

    if skip_if_exists and output_json.exists():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            text = _docagent_raw_text(data)
            elapsed = data.get("_benchmark_elapsed", 0.0)
            log.info("DocumentAgent: loaded cached output for %s", stem)
            return text, elapsed
        except Exception:
            pass

    cmd = [
        sys.executable, "-m", "src.document_agent.cli",
        "--input", str(pdf_path),
        "--output", str(output_json),
        "--max-pages", str(max_pages) if max_pages > 0 else "0",
        "--no-report",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900,
            cwd=str(Path(__file__).parent.parent),
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            log.warning("DocumentAgent failed for %s:\n%s", stem, proc.stderr[-500:])
            return f"[docagent error: {proc.returncode}]", elapsed
    except subprocess.TimeoutExpired:
        return "[docagent timeout]", 0.0

    try:
        data = json.loads(output_json.read_text(encoding="utf-8"))
        # Cache timing
        data["_benchmark_elapsed"] = elapsed
        output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return _docagent_raw_text(data), elapsed
    except Exception as exc:
        return f"[docagent parse error: {exc}]", elapsed


# ── Main benchmark loop ────────────────────────────────────────────────────────

ENGINES = {
    "tesseract":     run_tesseract,
    "easyocr":       run_easyocr,
    "paddleocr":     run_paddleocr,
    "marker":        None,   # special signature
    "docling":       run_docling,
    "documentagent": None,   # special signature
}


def run_one(
    engine: str,
    pdf_path: Path,
    max_pages: int,
    output_dir: Path,
    gt_text: str,
) -> Dict[str, Any]:
    log.info("  [%s] running on %s …", engine, pdf_path.name)
    try:
        if engine == "marker":
            text, elapsed = run_marker(pdf_path, max_pages, output_dir)
        elif engine == "documentagent":
            text, elapsed = run_documentagent(pdf_path, max_pages, output_dir)
        else:
            fn = ENGINES[engine]
            text, elapsed = fn(pdf_path, max_pages)
    except Exception as exc:
        log.exception("Engine %s crashed on %s", engine, pdf_path.name)
        text, elapsed = f"[crash: {exc}]", 0.0

    metrics = compute_metrics(text, gt_text)
    # Determine actual page count used
    import fitz
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()
    n_pages = min(max_pages, total_pages) if max_pages > 0 else total_pages
    spp = round(elapsed / n_pages, 2) if n_pages > 0 else 0.0

    return {
        "engine":    engine,
        "doc":       pdf_path.stem,
        "n_pages":   n_pages,
        "cer":       metrics["cer"],
        "wer":       metrics["wer"],
        "elapsed_s": round(elapsed, 1),
        "s_per_page": spp,
        "chars":     len(text),
        "error":     text[:120] if text.startswith("[") else "",
    }


def format_table(rows: List[Dict]) -> str:
    """Format results as a markdown table."""
    if not rows:
        return ""
    headers = ["engine", "doc", "n_pages", "cer", "wer", "s/page", "elapsed_s"]
    col_w = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}

    def fmt_row(r):
        return "| " + " | ".join(str(r.get(h, "")).ljust(col_w[h]) for h in headers) + " |"

    sep = "|" + "|".join("-" * (col_w[h] + 2) for h in headers) + "|"
    lines = [fmt_row({h: h for h in headers}), sep]
    for r in rows:
        row_d = {**r, "s/page": r["s_per_page"]}
        lines.append(fmt_row(row_d))
    return "\n".join(lines)


def _load_existing_results(csv_path: Path) -> List[Dict]:
    """Load existing results from CSV for append mode."""
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields back
            for k in ("cer", "wer", "elapsed_s", "s_per_page"):
                try:
                    row[k] = float(row[k])
                except (ValueError, KeyError):
                    row[k] = -1.0
            for k in ("n_pages", "chars"):
                try:
                    row[k] = int(row[k])
                except (ValueError, KeyError):
                    row[k] = 0
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare OCR engines on the same documents.")
    parser.add_argument("--docs-dir",    default="data/test_docs",    help="Folder with PDFs")
    parser.add_argument("--output-dir",  default="output/engine_compare", help="Results output folder")
    parser.add_argument("--max-pages",   type=int, default=5,          help="Pages per doc (0=all)")
    parser.add_argument("--engines",     default="all",               help="Comma-separated engine list or 'all'")
    parser.add_argument("--append",      action="store_true",          help="Append to existing results.csv instead of overwriting")
    parser.add_argument("--filter-docs", default="",                   help="Comma-separated doc stems to process (default: all)")
    parser.add_argument("--skip-docagent", action="store_true",        help="Skip re-running DocumentAgent if output exists")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence noisy libs
    for noisy in ("ppocr", "paddle", "PIL", "urllib3", "httpx", "easyocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    docs_dir   = Path(args.docs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve engines
    if args.engines.strip().lower() == "all":
        selected = list(ENGINES.keys())
    else:
        selected = [e.strip().lower() for e in args.engines.split(",")]
        unknown = [e for e in selected if e not in ENGINES]
        if unknown:
            log.error("Unknown engines: %s. Available: %s", unknown, list(ENGINES))
            sys.exit(1)

    # Find PDFs
    all_pdfs = sorted(docs_dir.glob("*.pdf"))
    if not all_pdfs:
        log.error("No PDFs found in %s", docs_dir)
        sys.exit(1)
    if args.filter_docs:
        filter_set = {s.strip() for s in args.filter_docs.split(",")}
        pdf_files = [p for p in all_pdfs if p.stem in filter_set]
        log.info("Filtered to %d docs: %s", len(pdf_files), [p.stem for p in pdf_files])
    else:
        pdf_files = all_pdfs

    log.info("Documents: %d | Engines: %s | Max pages: %s",
             len(pdf_files), selected, args.max_pages or "all")

    csv_path = output_dir / "results.csv"
    # Load existing results if appending
    existing: List[Dict] = []
    if args.append:
        existing = _load_existing_results(csv_path)
        # Build set of (engine, doc) already done to skip
        already_done = {(r["engine"], r["doc"]) for r in existing}
        log.info("Append mode: %d existing results loaded", len(existing))
    else:
        already_done = set()

    all_results: List[Dict] = list(existing)
    skipped_docs: List[str] = []

    for pdf_path in pdf_files:
        log.info("=== %s ===", pdf_path.name)

        # Ground truth
        gt_text, n_gt_pages = extract_gt_text(pdf_path, args.max_pages)
        if not _is_born_digital(gt_text, n_gt_pages):
            log.warning("  Skipping %s — sparse text layer (scanned?)", pdf_path.name)
            skipped_docs.append(pdf_path.stem)
            continue

        log.info("  Ground truth: %d chars across %d pages", len(gt_text), n_gt_pages)

        for engine in selected:
            if (engine, pdf_path.stem) in already_done:
                log.info("  [%s] skipping %s (already in results)", engine, pdf_path.stem)
                continue
            result = run_one(engine, pdf_path, args.max_pages, output_dir, gt_text)
            all_results.append(result)
            status = f"CER={result['cer']:.3f} WER={result['wer']:.3f}" if result["cer"] >= 0 else result["error"]
            log.info("  [%s] done — %s  (%ss)", engine, status, result["elapsed_s"])

    # ── Save results ─────────────────────────────────────────────────────────

    # CSV
    fieldnames = ["engine", "doc", "n_pages", "cer", "wer", "elapsed_s", "s_per_page", "chars", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    log.info("CSV → %s", csv_path)

    # JSON
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    md_lines = [
        "# OCR Engine Comparison\n",
        f"**Documents:** {len(pdf_files)} PDFs, **Max pages:** {args.max_pages or 'all'}\n",
    ]
    if skipped_docs:
        md_lines.append(f"**Skipped (sparse text layer):** {', '.join(skipped_docs)}\n")

    md_lines.append("\n## Results\n")
    md_lines.append(format_table(all_results))

    # Per-engine averages
    md_lines.append("\n## Averages per engine\n")
    valid = [r for r in all_results if r["cer"] >= 0]
    by_engine: Dict[str, List] = {}
    for r in valid:
        by_engine.setdefault(r["engine"], []).append(r)

    avg_rows = []
    for engine, rows in sorted(by_engine.items()):
        avg_rows.append({
            "engine": engine,
            "docs": len(rows),
            "avg_cer": round(sum(r["cer"] for r in rows) / len(rows), 4),
            "avg_wer": round(sum(r["wer"] for r in rows) / len(rows), 4),
            "avg_s_page": round(sum(r["s_per_page"] for r in rows) / len(rows), 2),
        })
    if avg_rows:
        avg_headers = ["engine", "docs", "avg_cer", "avg_wer", "avg_s_page"]
        col_w = {h: max(len(h), max(len(str(r[h])) for r in avg_rows)) for h in avg_headers}
        sep = "|" + "|".join("-" * (col_w[h] + 2) for h in avg_headers) + "|"
        md_lines.append("| " + " | ".join(h.ljust(col_w[h]) for h in avg_headers) + " |")
        md_lines.append(sep)
        for r in avg_rows:
            md_lines.append("| " + " | ".join(str(r[h]).ljust(col_w[h]) for h in avg_headers) + " |")

    md_path = output_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Report → %s", md_path)

    print(f"\n{'='*60}")
    print(f"  Results: {len(all_results)} runs across {len(pdf_files)} docs")
    print(f"  CSV:     {csv_path}")
    print(f"  Report:  {md_path}")
    print(f"{'='*60}\n")

    if avg_rows:
        print("Averages per engine:")
        for r in sorted(avg_rows, key=lambda x: x["avg_cer"]):
            print(f"  {r['engine']:15s}  CER={r['avg_cer']:.4f}  WER={r['avg_wer']:.4f}  {r['avg_s_page']:.1f}s/page")


if __name__ == "__main__":
    main()
