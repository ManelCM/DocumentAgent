from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import build_graph


def parse_args():
    parser = argparse.ArgumentParser(description="OCR multimodal estructural con LangGraph.")
    parser.add_argument("--input", required=True, help="Ruta del PDF o imagen.")
    parser.add_argument("--output", required=True, help="Ruta de salida JSON.")
    return parser.parse_args()


def main():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    args = parse_args()
    app = build_graph()
    result = app.invoke({"input_path": args.input, "status": "init"})
    output = result.get("output", {})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Structured output saved at: {out_path}")


if __name__ == "__main__":
    main()
