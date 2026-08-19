"""Test 2.2 — Intelligibility: ASR Round-Trip WER."""
from __future__ import annotations

import gc
import logging
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_intelligibility_sentences
from eval.tts.client_factory import build_tts_client
from eval.utils import NORMALIZE_FOR_WER, get_completed_ids, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.intelligibility")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
#
# The reference sentences for the `numbers` and `technical` categories spell
# digits and symbols in words ("one thousand two hundred and thirty-four
# dollars and fifty-six cents"), while Whisper transcribes the synthesised
# audio back in digit form ("$1,234.56"). Plain jiwer normalisation lowercases
# and strips punctuation but does not unify the two surface forms, which
# inflated WER on those categories to 25–54 % even though the audio is
# perfectly intelligible.
#
# Whisper's official EnglishTextNormalizer (used in the Whisper paper for the
# same problem) handles digit↔word, currency, contractions, and unit
# normalisation. We apply it to both sides before jiwer so the metric reflects
# semantic equality, not surface mismatch.
def _build_english_normalizer():
    try:
        from whisper.normalizers import EnglishTextNormalizer
        return EnglishTextNormalizer()
    except Exception as exc:  # pragma: no cover — fallback path
        logger.warning(
            "Whisper EnglishTextNormalizer unavailable (%s); falling back to "
            "jiwer-only normalisation. WER on numbers/technical will be "
            "inflated by surface form mismatches.",
            exc,
        )
        return None


def _normalize_pair(normalizer, ref: str, hyp: str) -> tuple[str, str]:
    if normalizer is None:
        return ref, hyp
    return normalizer(ref), normalizer(hyp)


def _transcribe_whisper(audio_path: str, model=None):
    """Transcribe audio using Whisper large-v3."""
    import whisper
    if model is None:
        model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path)
    return result["text"]


def _seed_categories_from_jsonl(jsonl_path: Path) -> dict[str, dict]:
    """Reconstruct per-category ref/hyp lists from existing JSONL.

    Resume-safe: when every sentence was completed on a prior run, the loop
    appends nothing and the summary would be empty otherwise.
    """
    seeded: dict[str, dict] = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in read_jsonl(jsonl_path):
        cat = r.get("category")
        ref = r.get("original_text")
        hyp = r.get("whisper_transcript")
        if cat and ref is not None and hyp is not None:
            seeded[cat]["refs"].append(ref)
            seeded[cat]["hyps"].append(hyp)
    return seeded


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "intelligibility"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = build_tts_client(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.2: Intelligibility (Round-Trip WER) ===")

    categories = get_intelligibility_sentences()
    logger.info("Categories: %s", list(categories.keys()))

    # Load Whisper once
    import whisper
    logger.info("Loading Whisper large-v3...")
    whisper_model = whisper.load_model("large-v3")

    completed = get_completed_ids(jsonl_path)
    seeded = _seed_categories_from_jsonl(jsonl_path)
    summary_rows = []

    normalizer = _build_english_normalizer()
    if normalizer is not None:
        logger.info("Using Whisper EnglishTextNormalizer for WER computation")

    for category, sentences in categories.items():
        logger.info("Category: %s (%d sentences)", category, len(sentences))
        # Seed with anything already on disk so resume-only runs still produce
        # a summary row.
        refs = list(seeded[category]["refs"])
        hyps = list(seeded[category]["hyps"])
        per_sent_wers = []

        for i, text in enumerate(tqdm(sentences, desc=category)):
            item_id = f"{category}_{i}"
            if item_id in completed:
                continue
            wav_path = results_dir / f"synth_{category}_{i:04d}.wav"

            try:
                result = tts_client.save_synthesis(text, str(wav_path))
                hyp = _transcribe_whisper(str(wav_path), whisper_model)
                wav_path.unlink(missing_ok=True)

                ref_n, hyp_n = _normalize_pair(normalizer, text, hyp)
                wer_val = jiwer.wer(
                    ref_n, hyp_n,
                    reference_transform=NORMALIZE_FOR_WER,
                    hypothesis_transform=NORMALIZE_FOR_WER,
                )
                cer_val = jiwer.cer(
                    ref_n, hyp_n,
                    reference_transform=NORMALIZE_FOR_WER,
                    hypothesis_transform=NORMALIZE_FOR_WER,
                )

                refs.append(text)
                hyps.append(hyp)
                per_sent_wers.append(wer_val)

                write_jsonl(jsonl_path, {
                    "id": f"{category}_{i}",
                    "category": category,
                    "original_text": text,
                    "whisper_transcript": hyp,
                    "normalized_reference": ref_n,
                    "normalized_hypothesis": hyp_n,
                    "wer": wer_val,
                    "cer": cer_val,
                })
            except Exception as e:
                wav_path.unlink(missing_ok=True)
                logger.warning("Failed %s_%d: %s", category, i, e)

        if refs:
            # Aggregate WER over the whole category — apply the same normaliser
            # so the per-sentence and aggregate metrics agree.
            refs_n = [normalizer(r) for r in refs] if normalizer else refs
            hyps_n = [normalizer(h) for h in hyps] if normalizer else hyps
            agg_wer = jiwer.wer(refs_n, hyps_n, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)
            agg_cer = jiwer.cer(refs_n, hyps_n, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)

            row = {
                "category": category,
                "n_sentences": len(refs),
                "round_trip_wer": round(agg_wer, 4),
                "round_trip_cer": round(agg_cer, 4),
                "normalizer": "whisper_english" if normalizer else "jiwer_only",
            }
            summary_rows.append(row)
            logger.info("  %s: WER=%.2f%% CER=%.2f%%", category, agg_wer * 100, agg_cer * 100)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    # Delete Whisper model cache — large (~3 GB) and not needed after this test.
    logger.info("Releasing Whisper model and cleaning cache...")
    del whisper_model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    whisper_cache = Path.home() / ".cache" / "whisper"
    for f in sorted(whisper_cache.glob("large-v3*")):
        try:
            f.unlink()
            logger.info("Deleted Whisper cache file: %s", f)
        except Exception:
            pass

    return {"test": "2.2", "name": "intelligibility", "results": summary_rows}
