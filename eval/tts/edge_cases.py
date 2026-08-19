"""Test 2.7 — Edge Cases & Input Robustness."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import jiwer
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_edge_cases
from eval.tts.client_factory import build_tts_client
from eval.tts.utmos import score_utmos
from eval.utils import NORMALIZE_FOR_WER, get_completed_ids, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.edge_cases")

# Edge cases stress-test the engine. KugelAudio has no documented SSML support,
# so the SSML/markup cases are treated as plain-text robustness probes — a pass
# means the tag characters were handled without erroring, not that markup was
# interpreted. Each call is bounded by config.tts.kugel.request_timeout_s so a
# single bad input can't freeze the whole test.


def _round_trip_wer(text: str, audio_path: str) -> float | None:
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return float(jiwer.wer(
            text, result["text"],
            reference_transform=NORMALIZE_FOR_WER,
            hypothesis_transform=NORMALIZE_FOR_WER,
        ))
    except Exception:
        return None


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "edge_cases"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = build_tts_client(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.7: Edge Cases & Input Robustness ===")

    edge_cases = get_edge_cases()
    logger.info("Testing %d edge cases", len(edge_cases))

    completed = get_completed_ids(jsonl_path)
    summary_rows = []

    # Seed summary_rows from existing JSONL so a resume-only run still has
    # something to aggregate. Records skipped via `continue` below would
    # otherwise leave summary_rows empty and the per-category aggregator at
    # the end would write a near-empty CSV.
    for r in read_jsonl(jsonl_path):
        if "category" in r and "status" in r:
            summary_rows.append(r)

    for i, case in enumerate(tqdm(edge_cases, desc="Edge cases")):
        item_id = f"edge_{i}"
        if item_id in completed:
            continue
        text = case["text"]
        category = case["category"]
        wav_path = results_dir / f"edge_{i:04d}.wav"

        record = {
            "id": f"edge_{i}",
            "category": category,
            "text": text[:200],
            "status": None,
            "error": None,
            "utmos": None,
            "round_trip_wer": None,
            "audio_duration_s": None,
        }

        try:
            result = tts_client.save_synthesis(text, str(wav_path))
            record["status"] = "pass"
            record["audio_duration_s"] = result["audio_duration"]

            if result["audio_duration"] > 0.1:
                record["utmos"] = score_utmos(str(wav_path))
                record["round_trip_wer"] = _round_trip_wer(text, str(wav_path))
            else:
                record["status"] = "degraded"
                record["error"] = "audio too short"
        except TimeoutError as e:
            # Request deadline exceeded — server hung on this input. Log loudly
            # so we notice in run.log, but keep moving so the test completes.
            record["status"] = "timeout"
            record["error"] = str(e)
            logger.warning("Edge case %d (%s) timed out: %s", i, category, e)
        except Exception as e:
            record["status"] = "fail"
            record["error"] = str(e)
        finally:
            wav_path.unlink(missing_ok=True)

        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # Aggregate by category
    category_summary = {}
    for row in summary_rows:
        cat = row["category"]
        if cat not in category_summary:
            category_summary[cat] = {"total": 0, "pass": 0, "fail": 0, "degraded": 0, "timeout": 0}
        category_summary[cat]["total"] += 1
        category_summary[cat][row["status"] or "fail"] += 1

    agg_rows = []
    for cat, counts in sorted(category_summary.items()):
        agg_rows.append({
            "category": cat,
            **counts,
            "pass_rate": counts["pass"] / counts["total"] if counts["total"] > 0 else 0,
        })

    save_summary_csv(results_dir / "summary.csv", agg_rows)

    total = len(summary_rows)
    passed = sum(1 for r in summary_rows if r["status"] == "pass")
    failed = sum(1 for r in summary_rows if r["status"] == "fail")
    timed_out = sum(1 for r in summary_rows if r["status"] == "timeout")
    logger.info("Edge cases: %d/%d passed, %d failed, %d timed out", passed, total, failed, timed_out)

    return {"test": "2.7", "name": "edge_cases", "results": agg_rows}
