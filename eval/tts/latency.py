"""Test 2.5 — Streaming TTFB & RTF (WebSocket vs REST)."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_text_length_buckets
from eval.tts.client_factory import build_tts_client
from eval.utils import compute_percentiles, get_completed_ids, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.latency")

N_REPS_PER_BUCKET = 50

# KugelAudio hard input limit (chars). Buckets longer than this are expected to
# fail with HTTP 400; we log that rather than treating it as an outage.
MAX_INPUT_CHARS = 10_000


def _seed_buckets_from_jsonl(jsonl_path: Path) -> dict[tuple, dict]:
    """Reconstruct per-(bucket, interface) accumulators from existing JSONL.

    Resume-safe: if every rep was completed in a prior session, the in-loop
    accumulators stay empty and the summary write is skipped — losing the
    aggregated CSV. Pre-seeding from disk fixes that without re-running the
    loop.
    """
    seeded: dict[tuple, dict] = defaultdict(
        lambda: {"ttfb": [], "rtf": [], "elapsed": []}
    )
    for r in read_jsonl(jsonl_path):
        bucket = r.get("bucket")
        interface = r.get("interface")
        if not bucket or not interface:
            continue
        key = (bucket, interface)
        if r.get("ttfb") is not None:
            seeded[key]["ttfb"].append(float(r["ttfb"]))
        if r.get("rtf") is not None:
            seeded[key]["rtf"].append(float(r["rtf"]))
        if r.get("elapsed_s") is not None:
            seeded[key]["elapsed"].append(float(r["elapsed_s"]))
    return seeded


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "latency"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = build_tts_client(config)
    jsonl_path = results_dir / "calls.jsonl"
    completed = get_completed_ids(jsonl_path)
    seeded = _seed_buckets_from_jsonl(jsonl_path)

    logger.info("=== Test 2.5: Streaming TTFB & RTF ===")

    buckets = get_text_length_buckets()

    summary_rows = []

    for bucket_name, texts in buckets.items():
        for interface in ["websocket", "rest"]:
            logger.info("Bucket=%s Interface=%s", bucket_name, interface)
            seeded_key = (bucket_name, interface)
            ttfb_values = list(seeded[seeded_key]["ttfb"])
            rtf_values = list(seeded[seeded_key]["rtf"])
            elapsed_values = list(seeded[seeded_key]["elapsed"])

            for rep in tqdm(range(N_REPS_PER_BUCKET), desc=f"{bucket_name}/{interface}"):
                item_id = f"{bucket_name}_{interface}_{rep}"
                if item_id in completed:
                    continue
                text = texts[rep % len(texts)]

                try:
                    if interface == "websocket":
                        result = tts_client.synthesize_stream(text)
                    else:
                        result = tts_client.synthesize_stream_rest(text)

                    ttfb = result.get("ttfb")
                    rtf = result.get("rtf")
                    elapsed = result.get("elapsed_s")

                    if ttfb is not None:
                        ttfb_values.append(ttfb)
                    if rtf is not None:
                        rtf_values.append(rtf)
                    if elapsed is not None:
                        elapsed_values.append(elapsed)

                    write_jsonl(jsonl_path, {
                        "id": f"{bucket_name}_{interface}_{rep}",
                        "bucket": bucket_name,
                        "interface": interface,
                        "text_len": len(text),
                        "ttfb": ttfb,
                        "rtf": rtf,
                        "elapsed_s": elapsed,
                        "audio_duration_s": result.get("audio_duration"),
                    })
                except Exception as e:
                    logger.warning("Failed %s/%s rep %d: %s", bucket_name, interface, rep, e)

            if ttfb_values:
                row = {
                    "bucket": bucket_name,
                    "interface": interface,
                    "n_calls": len(ttfb_values),
                    "ttfb": compute_percentiles(ttfb_values),
                    "rtf": compute_percentiles(rtf_values) if rtf_values else {},
                    "elapsed": compute_percentiles(elapsed_values) if elapsed_values else {},
                }
                summary_rows.append(row)
                logger.info("  TTFB P50=%.0fms P99=%.0fms, RTF P50=%.3f",
                    row["ttfb"].get("p50", 0) * 1000,
                    row["ttfb"].get("p99", 0) * 1000,
                    row["rtf"].get("p50", 0) if row["rtf"] else 0,
                )
            else:
                sample_len = len(texts[0]) if texts else 0
                if sample_len > MAX_INPUT_CHARS:
                    logger.warning(
                        "  Bucket %s/%s: all %d reps failed — text length %d chars exceeds "
                        "Kugel input limit (%d). Skipping bucket.",
                        bucket_name, interface, N_REPS_PER_BUCKET, sample_len, MAX_INPUT_CHARS,
                    )
                else:
                    logger.warning("  Bucket %s/%s: no successful calls", bucket_name, interface)

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "2.5", "name": "streaming_ttfb_rtf", "results": summary_rows}
