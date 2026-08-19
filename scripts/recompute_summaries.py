"""Recompute summary.csv files from existing JSONL records.

Fixes the 'all-resumed run wrote no summary' problem: if every JSONL record
for a test was already in place from a previous session, the test's run() may
exit without aggregating the in-memory lists into summary.csv. This script
walks the JSONL on disk and rebuilds the missing summaries directly.

Currently rebuilds:
  - tts/latency/summary.csv         (per-bucket × interface percentiles)
  - tts/intelligibility/summary.csv (per-category WER/CER)
  - tts/edge_cases/summary.csv      (per-category pass/fail/timeout counts)

Usage:
    python scripts/recompute_summaries.py [--results-dir results/run-09-06-26]

Idempotent — safe to re-run. Touches only summary.csv files; JSONLs untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _percentiles(values: list[float]) -> dict:
    """Mirror eval/utils.py compute_percentiles() output schema."""
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  no rows to write for {path}")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {len(rows)} rows → {path}")


def recompute_latency(test_dir: Path) -> None:
    print(f"[latency] {test_dir}")
    records = _read_jsonl(test_dir / "calls.jsonl")
    if not records:
        print("  no records, skipping")
        return

    buckets: dict[tuple, dict] = defaultdict(
        lambda: {"ttfb": [], "rtf": [], "elapsed": []}
    )
    for r in records:
        key = (r.get("bucket"), r.get("interface"))
        if key[0] is None or key[1] is None:
            continue
        for src, dst in [("ttfb", "ttfb"), ("rtf", "rtf"), ("elapsed_s", "elapsed")]:
            v = r.get(src)
            if v is not None:
                buckets[key][dst].append(float(v))

    rows = []
    for (bucket, interface), data in sorted(buckets.items()):
        if not data["ttfb"]:
            continue
        rows.append({
            "bucket": bucket,
            "interface": interface,
            "n_calls": len(data["ttfb"]),
            "ttfb": _percentiles(data["ttfb"]),
            "rtf": _percentiles(data["rtf"]),
            "elapsed": _percentiles(data["elapsed"]),
        })
    _write_csv(test_dir / "summary.csv", rows)


def recompute_intelligibility(test_dir: Path) -> None:
    print(f"[intelligibility] {test_dir}")
    records = _read_jsonl(test_dir / "calls.jsonl")
    if not records:
        print("  no records, skipping")
        return

    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "unknown")].append(r)

    rows = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        # Prefer pre-normalised wer_norm if present (recompute_intelligibility_wer
        # writes it), otherwise fall back to the raw 'wer' field.
        wer_field = "wer_norm" if "wer_norm" in items[0] else "wer"
        cer_field = "cer_norm" if "cer_norm" in items[0] else "cer"
        wers = [float(r[wer_field]) for r in items if r.get(wer_field) is not None]
        cers = [float(r[cer_field]) for r in items if r.get(cer_field) is not None]
        rows.append({
            "category": cat,
            "n_sentences": len(items),
            "round_trip_wer": round(float(np.mean(wers)), 4) if wers else None,
            "round_trip_cer": round(float(np.mean(cers)), 4) if cers else None,
        })
    _write_csv(test_dir / "summary.csv", rows)


def recompute_edge_cases(test_dir: Path) -> None:
    print(f"[edge_cases] {test_dir}")
    records = _read_jsonl(test_dir / "calls.jsonl")
    if not records:
        print("  no records, skipping")
        return

    by_cat: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "pass": 0, "fail": 0, "degraded": 0, "timeout": 0}
    )
    for r in records:
        cat = r.get("category", "unknown")
        status = r.get("status") or "fail"
        by_cat[cat]["total"] += 1
        if status not in by_cat[cat]:
            # Be tolerant of older schema variants without 'timeout'.
            by_cat[cat][status] = 0
        by_cat[cat][status] += 1

    rows = []
    for cat in sorted(by_cat):
        c = by_cat[cat]
        rows.append({
            "category": cat,
            **c,
            "pass_rate": round(c["pass"] / c["total"], 4) if c["total"] else 0,
        })
    _write_csv(test_dir / "summary.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()

    base = Path(args.results_dir)
    if not base.exists():
        raise SystemExit(f"results dir not found: {base}")

    recompute_latency(base / "tts" / "latency")
    recompute_intelligibility(base / "tts" / "intelligibility")
    recompute_edge_cases(base / "tts" / "edge_cases")

    print("\nDone. Re-run the report generator to pick these up:")
    print(f"  python -m eval.report.generate_report --results-dir {base} "
          f"--output {base}/report.html")


if __name__ == "__main__":
    main()
