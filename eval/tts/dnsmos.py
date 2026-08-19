"""Microsoft DNSMOS scorer — lazy-initialised, downloads ONNX models on first use."""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger("eval.tts.dnsmos")

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

_MODELS_BASE = (
    "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS"
)
_MODEL_FILES = {
    "sig_bak_ovr.onnx": f"{_MODELS_BASE}/sig_bak_ovr.onnx",
    "model_v8.onnx": f"{_MODELS_BASE}/model_v8.onnx",
}

_dnsmos_scorer = None  # ComputeScore instance or False
_model_dir: Path | None = None


def _get_model_dir() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = cache / "dnsmos_onnx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_models(model_dir: Path) -> bool:
    for fname, url in _MODEL_FILES.items():
        dest = model_dir / fname
        if dest.exists():
            continue
        logger.info("Downloading DNSMOS model %s …", fname)
        try:
            urllib.request.urlretrieve(url, dest)
            logger.info("Downloaded %s (%.1f MB)", fname, dest.stat().st_size / 1e6)
        except Exception as e:
            logger.warning("Failed to download DNSMOS model %s: %s", fname, e)
            return False
    return True


class _ComputeScore:
    def __init__(self, primary_model_path: str, p808_model_path: str) -> None:
        import onnxruntime as ort
        self._sess = ort.InferenceSession(primary_model_path)
        self._p808_sess = ort.InferenceSession(p808_model_path)

    def _audio_melspec(
        self, audio: np.ndarray, n_mels: int = 120, frame_size: int = 320,
        hop_length: int = 160, sr: int = 16000,
    ) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1,
            hop_length=hop_length, n_mels=n_mels,
        )
        mel_db = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        return mel_db.T

    def _polyfit(self, sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))

    def score(self, fpath: str) -> dict:
        aud, input_fs = sf.read(fpath)
        if aud.ndim > 1:
            aud = aud.mean(axis=1)
        audio = aud.astype(np.float32)
        if input_fs != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=input_fs, target_sr=SAMPLING_RATE)

        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len = SAMPLING_RATE

        sig_vals, bak_vals, ovr_vals = [], [], []
        for idx in range(num_hops):
            seg = audio[int(idx * hop_len): int((idx + INPUT_LENGTH) * hop_len)]
            if len(seg) < len_samples:
                continue
            inp = seg[np.newaxis, :].astype("float32")
            mel_inp = self._audio_melspec(seg[:-160])[np.newaxis, :, :].astype("float32")
            raw_sig, raw_bak, raw_ovr = self._sess.run(None, {"input_1": inp})[0][0]
            s, b, o = self._polyfit(raw_sig, raw_bak, raw_ovr)
            sig_vals.append(s)
            bak_vals.append(b)
            ovr_vals.append(o)

        if not sig_vals:
            return {}
        return {
            "ovrl": float(np.mean(ovr_vals)),
            "sig": float(np.mean(sig_vals)),
            "bak": float(np.mean(bak_vals)),
        }


def get_dnsmos_scorer() -> _ComputeScore | None:
    global _dnsmos_scorer, _model_dir
    if _dnsmos_scorer is False:
        return None
    if _dnsmos_scorer is not None:
        return _dnsmos_scorer

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.warning("onnxruntime not installed — DNSMOS unavailable")
        _dnsmos_scorer = False
        return None

    model_dir = _get_model_dir()
    if not _ensure_models(model_dir):
        _dnsmos_scorer = False
        return None

    try:
        _dnsmos_scorer = _ComputeScore(
            str(model_dir / "sig_bak_ovr.onnx"),
            str(model_dir / "model_v8.onnx"),
        )
        _model_dir = model_dir
        logger.info("DNSMOS scorer initialised (Microsoft ONNX, models at %s)", model_dir)
        return _dnsmos_scorer
    except Exception as e:
        logger.warning("DNSMOS init failed: %s", e)
        _dnsmos_scorer = False
        return None


def score_dnsmos(audio_path: str) -> dict | None:
    scorer = get_dnsmos_scorer()
    if scorer is None:
        return None
    try:
        result = scorer.score(audio_path)
        return result if result else None
    except Exception as e:
        logger.warning("DNSMOS scoring failed: %s", e)
        return None
