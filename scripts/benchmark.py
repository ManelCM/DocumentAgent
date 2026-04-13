"""
Benchmark runner for DocumentAgent.

Downloads a curated set of diverse PDFs from public sources,
runs the OCR pipeline on each, and prints a comparison table.

Usage:
    python scripts/benchmark.py [--skip-download] [--max-pages N]

Outputs:
    data/test_docs/<name>.pdf          — downloaded documents
    output/benchmark/<name>/output.json
    output/benchmark/summary.json      — aggregated metrics
    output/benchmark/report.md         — human-readable comparison table
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── Curated test corpus ────────────────────────────────────────────────────────
#
# Coverage matrix:
#   columns:  single-col | 2-col | wide-page
#   content:  math | tables | charts | images | formulas
#   quality:  born-digital | scanned (TFG example)
#
DOCS: List[Dict] = [
    {
        "name": "attention_all_you_need",
        "url": "https://arxiv.org/pdf/1706.03762",
        "category": "2-col, math-heavy, attention diagrams, tables",
        "source": "arXiv:1706.03762",
    },
    {
        "name": "resnet",
        "url": "https://arxiv.org/pdf/1512.03385",
        "category": "2-col, bar charts, accuracy tables",
        "source": "arXiv:1512.03385",
    },
    {
        "name": "mistral_7b",
        "url": "https://arxiv.org/pdf/2310.06825",
        "category": "single-col, evaluation tables, short",
        "source": "arXiv:2310.06825",
    },
    {
        "name": "lora",
        "url": "https://arxiv.org/pdf/2106.09685",
        "category": "2-col, math, parameter tables",
        "source": "arXiv:2106.09685",
    },
    {
        "name": "vggnet",
        "url": "https://arxiv.org/pdf/1409.1556",
        "category": "2-col, architecture tables, charts",
        "source": "arXiv:1409.1556",
    },
    {
        "name": "detr",
        "url": "https://arxiv.org/pdf/2005.12872",
        "category": "2-col, figure-heavy, detection tables",
        "source": "arXiv:2005.12872",
    },
    {
        "name": "clip",
        "url": "https://arxiv.org/pdf/2103.00020",
        "category": "single-col, many figures, benchmark tables",
        "source": "arXiv:2103.00020",
    },
]

# Documents already present in data/ — included in comparison but not downloaded
LOCAL_DOCS: List[Dict] = [
    {
        "name": "paper1",
        "path": "data/paper1.pdf",
        "category": "2-col, formulas, charts, tables (existing)",
        "source": "local",
    },
    {
        "name": "tfg_ocr_example",
        "path": "data/tfg_ocr_example.pdf",
        "category": "1-page, mixed content, OCR quality test (existing)",
        "source": "local",
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def download_pdf(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a PDF to dest. Returns True on success."""
    import urllib.request
    import urllib.error

    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [skip] {dest.name} already downloaded")
        return True

    print(f"  Downloading {url} …", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DocumentAgent-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"OK ({len(data) // 1024} KB)")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def run_pipeline(pdf_path: Path, output_dir: Path, max_pages: int = 30) -> Optional[Dict]:
    """Run the DocumentAgent pipeline on one PDF. Returns parsed output.json or None."""
    output_json = output_dir / "output.json"
    if output_json.exists():
        print(f"  [cached] {pdf_path.name} — reusing existing output")
        with open(output_json, encoding="utf-8") as f:
            return json.load(f)

    print(f"  Running pipeline on {pdf_path.name} …", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "src.document_agent.cli",
        "--input", str(pdf_path),
        "--output", str(output_json),
        "--no-report",
    ]
    if max_pages and max_pages > 0:
        cmd += ["--max-pages", str(max_pages)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        # Print last 20 lines of stderr for diagnosis
        lines = result.stderr.strip().splitlines()
        print(f"  ERROR (exit {result.returncode}) in {elapsed:.1f}s")
        for line in lines[-20:]:
            print(f"    {line}")
        return None

    print(f"  Done in {elapsed:.1f}s")
    with open(output_json, encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(doc_meta: Dict, data: Dict) -> Dict:
    s = data.get("summary", {})
    blocks = data.get("blocks", [])

    confs = [b["payload"]["ocr_confidence"] for b in blocks
             if b.get("payload", {}).get("ocr_confidence") is not None]
    avg_conf = round(sum(confs) / len(confs), 3) if confs else None

    formulas = [b for b in blocks if b.get("type") == "formula"]
    latex_ok = sum(1 for b in formulas if b.get("payload", {}).get("latex"))

    total_refs = sum(len(b.get("relations", {}).get("references", [])) for b in blocks)
    resolved = sum(
        1 for b in blocks
        for r in b.get("relations", {}).get("references", [])
        if r.get("target_id")
    )

    types = s.get("types", {})
    return {
        "name":          doc_meta["name"],
        "source":        doc_meta.get("source", ""),
        "category":      doc_meta.get("category", ""),
        "pages":         s.get("num_pages", 0),
        "blocks":        s.get("num_blocks", 0),
        "duration_s":    round(s.get("total_duration_s") or 0, 1),
        "tokens":        s.get("total_tokens", 0),
        "cost_usd":      s.get("total_cost_usd", 0),
        "llm_calls":     s.get("num_llm_calls", 0),
        "full_text_chars": s.get("full_text_chars", 0),
        "avg_ocr_conf":  avg_conf,
        "formula_total": len(formulas),
        "formula_latex": latex_ok,
        "xref_found":    total_refs,
        "xref_resolved": resolved,
        "types":         types,
        # per-type counts for quick scan
        "n_text":    types.get("text", 0),
        "n_table":   types.get("table", 0),
        "n_formula": types.get("formula", 0),
        "n_chart":   types.get("chart", 0),
        "n_image":   types.get("image", 0) + types.get("figure", 0),
    }


def print_table(rows: List[Dict]) -> str:
    cols = [
        ("Document",        "name",           20),
        ("Pages",           "pages",           5),
        ("Blocks",          "blocks",          6),
        ("Time(s)",         "duration_s",      7),
        ("Tokens",          "tokens",          8),
        ("Cost($)",         "cost_usd",        8),
        ("OCR conf",        "avg_ocr_conf",    9),
        ("Formulas",        "formula_total",   8),
        ("LaTeX%",          "formula_latex",   7),
        ("XRef res",        "xref_resolved",   8),
        ("Text",            "n_text",          5),
        ("Tbl",             "n_table",         4),
        ("Chrt",            "n_chart",         5),
        ("Img",             "n_image",         4),
    ]

    header = " | ".join(label.ljust(w) for label, _, w in cols)
    sep = "-+-".join("-" * w for _, _, w in cols)
    lines = [header, sep]

    for r in rows:
        def _fmt(key, w):
            v = r.get(key)
            if v is None:
                return "—".ljust(w)
            if key == "cost_usd":
                return f"${v:.4f}".ljust(w)
            if key == "avg_ocr_conf":
                return f"{v:.3f}".ljust(w)
            if key == "formula_latex":
                total = r.get("formula_total", 0)
                pct = f"{int(v/total*100)}%" if total else "—"
                return pct.ljust(w)
            return str(v).ljust(w)

        lines.append(" | ".join(_fmt(key, w) for _, key, w in cols))

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DocumentAgent multi-doc benchmark")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading; only run pipeline on existing files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip docs that already have an output.json")
    parser.add_argument("--max-pages", type=int, default=30,
                        help="Max pages to process per document (default 30, 0=all)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    doc_dir = root / "data" / "test_docs"
    out_root = root / "output" / "benchmark"
    doc_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    all_docs = []

    # ── Download remote docs ──────────────────────────────────────────────────
    if not args.skip_download:
        print("\n=== Downloading test documents ===")
        for meta in DOCS:
            dest = doc_dir / f"{meta['name']}.pdf"
            ok = download_pdf(meta["url"], dest)
            if ok:
                all_docs.append({**meta, "path": str(dest)})
            else:
                print(f"  Skipping {meta['name']} (download failed)")
    else:
        for meta in DOCS:
            dest = doc_dir / f"{meta['name']}.pdf"
            if dest.exists():
                all_docs.append({**meta, "path": str(dest)})

    # ── Add local docs ────────────────────────────────────────────────────────
    for meta in LOCAL_DOCS:
        p = root / meta["path"]
        if p.exists():
            all_docs.append({**meta, "path": str(p)})
        else:
            print(f"  [skip] local doc not found: {meta['path']}")

    # ── Run pipeline on each ──────────────────────────────────────────────────
    print(f"\n=== Running pipeline on {len(all_docs)} documents ===")
    metrics_rows = []

    for meta in all_docs:
        pdf_path = Path(meta["path"])
        out_dir = out_root / meta["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{meta['name']}] {meta['category']}")

        if args.skip_existing and (out_dir / "output.json").exists():
            print("  [skip] output already exists")
            with open(out_dir / "output.json", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = run_pipeline(pdf_path, out_dir, max_pages=args.max_pages)

        if data:
            metrics_rows.append(extract_metrics(meta, data))
        else:
            metrics_rows.append({
                "name": meta["name"],
                "category": meta.get("category", ""),
                "source": meta.get("source", ""),
                "error": True,
            })

    # ── Save + print summary ──────────────────────────────────────────────────
    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    good_rows = [r for r in metrics_rows if not r.get("error")]
    failed = [r["name"] for r in metrics_rows if r.get("error")]

    table_str = print_table(good_rows)

    report_lines = [
        "# DocumentAgent — Multi-Document Benchmark",
        "",
        "## Results",
        "",
        "```",
        table_str,
        "```",
        "",
    ]
    if failed:
        report_lines += ["## Failed", "", *[f"- {n}" for n in failed], ""]

    # Per-doc category notes
    report_lines += ["## Document index", ""]
    for r in good_rows:
        report_lines.append(f"- **{r['name']}** ({r['source']}): {r['category']}")

    report_md = "\n".join(report_lines)
    report_path = out_root / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print("\n\n=== BENCHMARK RESULTS ===\n")
    print(table_str)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
    print(f"\nSummary JSON: {summary_path}")
    print(f"Report MD:    {report_path}")


if __name__ == "__main__":
    main()
