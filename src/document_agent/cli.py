from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    logging.basicConfig(stream=stream, level=level, format=fmt, datefmt="%H:%M:%S")
    # Silence noisy third-party loggers
    for noisy in ("ppocr", "paddle", "PIL", "urllib3", "httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description="OCR multimodal estructural con LangGraph.")
    parser.add_argument("--input",   required=True,  help="Ruta del PDF o imagen.")
    parser.add_argument("--output",  required=True,  help="Ruta de salida JSON.")
    parser.add_argument("--report",  default=None,   help="Directorio donde generar el informe HTML/PNG. Si se omite se genera junto al output.")
    parser.add_argument("--no-report", action="store_true", help="Desactivar generación de informe.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logs de nivel DEBUG.")
    return parser.parse_args()


def main():
    # Must be set before any native library (PaddleOCR + LayoutDetection both use OpenMP;
    # concurrent initialisation without this causes a segfault on Windows).
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    args = parse_args()
    _setup_logging(args.verbose)
    log = logging.getLogger("document_agent.cli")

    from .graph import build_graph
    app = build_graph().compile()

    log.info("Starting OCR pipeline on: %s", args.input)
    result = app.invoke({"input_path": args.input, "status": "init"})
    output = result.get("output", {})
    trace  = result.get("_trace", {})
    pages  = result.get("pages", [])

    # ── Save structured output JSON ──────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Structured output → %s", out_path)

    # ── Generate report ──────────────────────────────────────────────────────
    if not args.no_report:
        report_dir = Path(args.report) if args.report else out_path.parent / (out_path.stem + "_report")
        try:
            from .visualizer import generate_report
            html = generate_report(output, trace, pages, report_dir)
            log.info("Report → %s", html)
            print(f"\nReport: {html}")
        except Exception as exc:
            log.warning("Report generation failed: %s", exc)

    total = trace.get("total_duration_s")
    n = output.get("summary", {}).get("num_blocks", 0)
    print(f"\nDone — {n} blocks processed" + (f" in {total:.1f}s" if total else "") + f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
