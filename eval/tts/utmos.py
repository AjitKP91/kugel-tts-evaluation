"""UTMOS22 scorer — lazy-initialised, one model per process.

Strategy: torch.hub tarepan/SpeechMOS (no fairseq dependency).
Falls back to unavailable if torch is missing or the download fails.
"""
from __future__ import annotations

import gc
import logging

import soundfile as sf

logger = logging.getLogger("eval.tts.utmos")

_utmos_scorer = None  # ("hub", model) | False
_utmos_available = None


def get_utmos_scorer():
    global _utmos_scorer, _utmos_available
    if _utmos_available is False:
        return None
    if _utmos_scorer is not None:
        return _utmos_scorer

    try:
        import torch
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        model.eval()
        _utmos_scorer = ("hub", model)
        _utmos_available = True
        logger.info("UTMOS scorer initialised (torch.hub tarepan/SpeechMOS:v1.2.0)")
        return _utmos_scorer
    except Exception as e:
        logger.warning("UTMOS (torch.hub) not available: %s", e)
        _utmos_available = False
        return None


def score_utmos(audio_path: str) -> float | None:
    result = get_utmos_scorer()
    if result is None:
        return None
    _, model = result
    try:
        import torch
        wave, sr = sf.read(audio_path, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave_tensor = torch.from_numpy(wave).unsqueeze(0)  # (1, T)
        with torch.no_grad():
            scores = model(wave_tensor, sr)
        return float(scores[0].item())
    except Exception as e:
        logger.warning("UTMOS scoring failed: %s", e)
    return None


def release_utmos() -> None:
    global _utmos_scorer, _utmos_available
    if _utmos_scorer is None:
        return
    _, model = _utmos_scorer
    del model
    _utmos_scorer = None
    _utmos_available = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
