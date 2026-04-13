"""
Quick helper: merge benchmark result CSVs and regenerate report.
Run this after all engine runs have completed.

Usage:
    python scripts/_merge_results.py [--output-dir output/engine_compare]
"""
import csv, sys
from pathlib import Path
from collections import defaultdict

def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/engine_compare")
    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv found at {csv_path}")
        return

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ("cer", "wer", "elapsed_s", "s_per_page"):
                try: row[k] = float(row[k])
                except: row[k] = -1.0
            for k in ("n_pages", "chars"):
                try: row[k] = int(row[k])
                except: row[k] = 0
            rows.append(row)

    # De-duplicate: keep latest (last) entry per (engine, doc)
    seen = {}
    for r in rows:
        seen[(r["engine"], r["doc"])] = r
    rows = list(seen.values())

    # Filter out error rows for summary (cer < 0 or cer > 0.95)
    valid = [r for r in rows if 0 <= r["cer"] < 0.95]
    errors = [r for r in rows if r["cer"] < 0 or r["cer"] >= 0.95]

    print(f"\nTotal results: {len(rows)} | Valid: {len(valid)} | Errors/empty: {len(errors)}")
    if errors:
        print("Error/skipped rows:")
        for r in errors:
            print(f"  {r['engine']:15} {r['doc']:30} CER={r['cer']:.3f}")

    # Per-engine averages (valid only)
    by_engine = defaultdict(list)
    for r in valid:
        by_engine[r["engine"]].append(r)

    print("\nAverages per engine (excluding error rows):")
    print(f"{'Engine':15} {'Docs':5} {'avg CER':8} {'avg WER':8} {'s/page':8}")
    print("-" * 50)
    for engine in sorted(by_engine, key=lambda e: sum(r["cer"] for r in by_engine[e])/len(by_engine[e])):
        rs = by_engine[engine]
        avg_cer = sum(r["cer"] for r in rs) / len(rs)
        avg_wer = sum(r["wer"] for r in rs) / len(rs)
        avg_spp = sum(r["s_per_page"] for r in rs) / len(rs)
        print(f"{engine:15} {len(rs):5} {avg_cer:8.4f} {avg_wer:8.4f} {avg_spp:8.1f}")

    # Save de-duplicated CSV
    out_csv = out_dir / "results_merged.csv"
    fieldnames = ["engine", "doc", "n_pages", "cer", "wer", "elapsed_s", "s_per_page", "chars", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMerged CSV → {out_csv}")

if __name__ == "__main__":
    main()
