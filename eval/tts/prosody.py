"""Test 2.3 — Prosody Analysis (reference-free)."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

try:
    import pyworld as pw
    _PYWORLD_AVAILABLE = True
except ImportError:
    pw = None
    _PYWORLD_AVAILABLE = False

from eval.config import Config
from eval.data.tts_test_sets import get_naturalness_sentences
from eval.tts.client_factory import build_tts_client
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.prosody")

_TEST_SENTENCES = [
    "The meeting has been rescheduled to Thursday afternoon at three o'clock.",
    "Please review the attached document before the end of the day.",
    "Our quarterly revenue exceeded expectations by twelve percent.",
    "Can you confirm receipt of the invoice we sent last week?",
    "The new policy takes effect on the first of next month.",
    "She asked whether the deadline could be extended by two days.",
    "Technical issues delayed the deployment by approximately six hours.",
    "We need to finalize the budget before the board meeting on Friday.",
    "All employees are required to complete the training by December.",
    "The project is currently on track and within budget.",
] * 5  # 50 sentences


def extract_f0(audio_path: str, sr: int = 16000) -> np.ndarray:
    if not _PYWORLD_AVAILABLE:
        return np.array([])
    wav, _ = librosa.load(audio_path, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr)
    f0 = pw.stonemask(wav, f0, t, sr)
    return f0[f0 > 0]  # voiced frames only


def speaking_rate_wpm(text: str, audio_path: str) -> float:
    duration = librosa.get_duration(path=audio_path)
    word_count = len(text.split())
    return (word_count / duration) * 60 if duration > 0 else 0.0


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "prosody"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = build_tts_client(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.3: Prosody (reference-free) ===")

    f0_means: list[float] = []
    wpms: list[float] = []
    all_records: list[dict] = []

    for i, text in enumerate(tqdm(_TEST_SENTENCES, desc="Prosody")):
        synth_path = results_dir / f"synth_{i:04d}.wav"
        try:
            tts_client.save_synthesis(text, str(synth_path))
            wpm = speaking_rate_wpm(text, str(synth_path))
            f0 = extract_f0(str(synth_path))

            mean_f0 = float(np.mean(f0)) if len(f0) > 0 else None
            if mean_f0 is not None:
                f0_means.append(mean_f0)
            wpms.append(wpm)

            record = {
                "id": f"prosody_{i}",
                "text": text,
                "wpm": round(wpm, 1),
                "syn_mean_f0": round(mean_f0, 1) if mean_f0 is not None else None,
            }
            write_jsonl(jsonl_path, record)
            all_records.append(record)
        except Exception as e:
            logger.warning("Failed prosody %d: %s", i, e)
        finally:
            synth_path.unlink(missing_ok=True)

    summary = {
        "f0_mean_hz": round(float(np.mean(f0_means)), 1) if f0_means else None,
        "f0_std_hz": round(float(np.std(f0_means)), 1) if f0_means else None,
        "f0_min_hz": round(float(np.min(f0_means)), 1) if f0_means else None,
        "f0_max_hz": round(float(np.max(f0_means)), 1) if f0_means else None,
        "speaking_rate_wpm_mean": round(float(np.mean(wpms)), 1) if wpms else None,
        "speaking_rate_wpm_std": round(float(np.std(wpms)), 1) if wpms else None,
        "n_sentences": len(_TEST_SENTENCES),
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    if f0_means:
        logger.info(
            "F0: mean=%.1f Hz  std=%.1f Hz  range=%.1f–%.1f Hz",
            np.mean(f0_means), np.std(f0_means), np.min(f0_means), np.max(f0_means),
        )
    if wpms:
        logger.info("Speaking rate: %.0f ± %.0f WPM", np.mean(wpms), np.std(wpms))

    return {"test": "2.3", "name": "prosody", "results": summary}
