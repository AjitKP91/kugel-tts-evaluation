"""Test 2.1 — Automated Naturalness (UTMOS / DNSMOS / NISQA)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_naturalness_sentences
from eval.tts.client_factory import build_tts_client
from eval.tts.dnsmos import score_dnsmos
from eval.tts.utmos import release_utmos, score_utmos
from eval.utils import compute_percentiles, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.naturalness")


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "naturalness"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = build_tts_client(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.1: Automated Naturalness ===")

    sentences = get_naturalness_sentences()
    logger.info("Evaluating %d sentences", len(sentences))

    # Load existing records; re-process any that are missing UTMOS scores.
    existing_records: dict[str, dict] = {}
    for r in read_jsonl(jsonl_path):
        if "id" in r:
            existing_records[r["id"]] = r

    utmos_null_ids = {id_ for id_, r in existing_records.items() if r.get("utmos") is None}
    fully_done = set(existing_records.keys()) - utmos_null_ids

    if utmos_null_ids:
        logger.info(
            "%d records missing UTMOS — will re-synthesise to score them "
            "(existing DNSMOS cached)",
            len(utmos_null_ids),
        )

    for i, text in enumerate(tqdm(sentences, desc="Naturalness")):
        item_id = f"nat_{i}"
        if item_id in fully_done:
            continue

        wav_path = results_dir / f"synth_{i:04d}.wav"
        try:
            result = tts_client.save_synthesis(text, str(wav_path))

            utmos = score_utmos(str(wav_path))

            # Re-use cached DNSMOS if this was a UTMOS-only re-score pass.
            existing = existing_records.get(item_id)
            if existing and existing.get("dnsmos_ovrl") is not None:
                dnsmos = {
                    "ovrl": existing["dnsmos_ovrl"],
                    "sig": existing["dnsmos_sig"],
                    "bak": existing["dnsmos_bak"],
                }
            else:
                dnsmos = score_dnsmos(str(wav_path))

            wav_path.unlink(missing_ok=True)

            record = {
                "id": item_id,
                "text": text,
                "audio_duration_s": result["audio_duration"],
                "utmos": utmos,
                "dnsmos_ovrl": dnsmos["ovrl"] if dnsmos else None,
                "dnsmos_sig": dnsmos["sig"] if dnsmos else None,
                "dnsmos_bak": dnsmos["bak"] if dnsmos else None,
            }
            write_jsonl(jsonl_path, record)
            existing_records[item_id] = record

        except Exception as e:
            logger.warning("Failed sentence %d: %s", i, e)

    # Deduplicate JSONL (keep latest record per ID) if we did any re-scoring.
    if utmos_null_ids:
        with jsonl_path.open("w") as f:
            for r in existing_records.values():
                f.write(json.dumps(r, default=str) + "\n")

    # Collect scores from all records.
    utmos_scores, dnsmos_ovrl_scores, dnsmos_sig_scores, dnsmos_bak_scores = [], [], [], []
    for rec in existing_records.values():
        if rec.get("utmos") is not None:
            utmos_scores.append(rec["utmos"])
        if rec.get("dnsmos_ovrl") is not None:
            dnsmos_ovrl_scores.append(rec["dnsmos_ovrl"])
            dnsmos_sig_scores.append(rec["dnsmos_sig"])
            dnsmos_bak_scores.append(rec["dnsmos_bak"])

    summary = {
        "n_sentences": len(sentences),
        "utmos": compute_percentiles(utmos_scores) if utmos_scores else {},
        "dnsmos_ovrl": compute_percentiles(dnsmos_ovrl_scores) if dnsmos_ovrl_scores else {},
        "dnsmos_sig": compute_percentiles(dnsmos_sig_scores) if dnsmos_sig_scores else {},
        "dnsmos_bak": compute_percentiles(dnsmos_bak_scores) if dnsmos_bak_scores else {},
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    if utmos_scores:
        logger.info("UTMOS: mean=%.3f std=%.3f", np.mean(utmos_scores), np.std(utmos_scores))
    if dnsmos_ovrl_scores:
        logger.info("DNSMOS OVRL: mean=%.3f", np.mean(dnsmos_ovrl_scores))

    release_utmos()

    return {"test": "2.1", "name": "naturalness", "results": summary}
