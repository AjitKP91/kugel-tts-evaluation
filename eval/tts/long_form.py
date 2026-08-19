"""Test 2.8 — Long-Form Voice Consistency."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyworld as pw
import librosa
import soundfile as sf
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_long_form_passages
from eval.tts.client_factory import build_tts_client
from eval.utils import compute_percentiles, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.long_form")


def _compute_speaker_embeddings(audio_paths: list[str]) -> np.ndarray:
    """Compute ECAPA-TDNN speaker embeddings for each audio file."""
    try:
        import torch
        if not hasattr(torch.amp, "custom_fwd"):
            def _custom_fwd_compat(fn=None, *, device_type=None, **kwargs):
                if fn is not None:
                    return torch.cuda.amp.custom_fwd(fn, **kwargs)  # type: ignore[attr-defined]
                if kwargs:
                    return torch.cuda.amp.custom_fwd(**kwargs)       # type: ignore[attr-defined]
                return torch.cuda.amp.custom_fwd                     # type: ignore[attr-defined]
            def _custom_bwd_compat(fn=None, **kwargs):
                if fn is not None:
                    return torch.cuda.amp.custom_bwd(fn, **kwargs)  # type: ignore[attr-defined]
                return torch.cuda.amp.custom_bwd                     # type: ignore[attr-defined]
            torch.amp.custom_fwd = _custom_fwd_compat                # type: ignore[attr-defined]
            torch.amp.custom_bwd = _custom_bwd_compat                # type: ignore[attr-defined]
        try:
            from speechbrain.inference.classifiers import EncoderClassifier  # >= 1.0
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore  # < 1.0

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
        )

        embeddings = []
        for path in audio_paths:
            wav, sr = sf.read(path, dtype="float32")
            wav_tensor = torch.tensor(wav).unsqueeze(0)
            emb = classifier.encode_batch(wav_tensor).squeeze().cpu().detach().numpy()
            embeddings.append(emb)

        return np.array(embeddings)
    except Exception as e:
        logger.warning("SpeechBrain not available: %s. Skipping speaker similarity.", e)
        return np.array([])


def _pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized @ normalized.T


def _extract_mean_f0(audio_path: str, sr: int = 16000) -> float:
    wav, actual_sr = librosa.load(audio_path, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr)
    f0 = pw.stonemask(wav, f0, t, sr)
    voiced = f0[f0 > 0]
    return float(np.mean(voiced)) if len(voiced) > 0 else 0.0


def _speaking_rate_wpm(text: str, audio_path: str) -> float:
    duration = librosa.get_duration(path=audio_path)
    word_count = len(text.split())
    return (word_count / duration) * 60 if duration > 0 else 0.0


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "long_form"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = build_tts_client(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.8: Long-Form Voice Consistency ===")

    passages = get_long_form_passages()
    completed = get_completed_ids(jsonl_path, id_field="passage_id")
    summary_rows = []

    for pi, passage in enumerate(tqdm(passages, desc="Passages")):
        if pi in completed:
            continue
        passage_dir = results_dir / f"passage_{pi}"
        passage_dir.mkdir(exist_ok=True)

        paragraph_paths = []
        paragraph_f0s = []
        paragraph_wpms = []

        for pj, paragraph in enumerate(passage["paragraphs"]):
            wav_path = passage_dir / f"para_{pj:02d}.wav"
            try:
                result = tts_client.save_synthesis(paragraph, str(wav_path))
                paragraph_paths.append(str(wav_path))

                mean_f0 = _extract_mean_f0(str(wav_path))
                wpm = _speaking_rate_wpm(paragraph, str(wav_path))
                paragraph_f0s.append(mean_f0)
                paragraph_wpms.append(wpm)

            except Exception as e:
                logger.warning("Passage %d para %d failed: %s", pi, pj, e)

        if len(paragraph_paths) < 2:
            continue

        # Speaker similarity
        embeddings = _compute_speaker_embeddings(paragraph_paths)
        sim_matrix = None
        mean_sim = None
        sim_std = None

        if len(embeddings) >= 2:
            sim_matrix = _pairwise_cosine_similarity(embeddings)
            triu_indices = np.triu_indices(len(embeddings), k=1)
            pairwise_sims = sim_matrix[triu_indices]
            mean_sim = float(np.mean(pairwise_sims))
            sim_std = float(np.std(pairwise_sims))

        f0_drift = float(np.std(paragraph_f0s)) if paragraph_f0s else None
        wpm_drift = float(np.std(paragraph_wpms)) if paragraph_wpms else None

        record = {
            "passage_id": pi,
            "title": passage.get("title", f"Passage {pi}"),
            "n_paragraphs": len(paragraph_paths),
            "mean_speaker_sim": round(mean_sim, 4) if mean_sim else None,
            "speaker_sim_std": round(sim_std, 4) if sim_std else None,
            "f0_drift_hz": round(f0_drift, 1) if f0_drift else None,
            "wpm_drift": round(wpm_drift, 1) if wpm_drift else None,
            "paragraph_f0s": [round(f, 1) for f in paragraph_f0s],
            "paragraph_wpms": [round(w, 1) for w in paragraph_wpms],
        }
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

        # WAVs no longer needed — delete them and the passage directory
        for p in paragraph_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            passage_dir.rmdir()
        except Exception:
            pass

        logger.info(
            "Passage %d: SpeakerSim=%.3f±%.3f, F0 drift=%.1f Hz, WPM drift=%.1f",
            pi,
            mean_sim or 0, sim_std or 0,
            f0_drift or 0, wpm_drift or 0,
        )

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "2.8", "name": "long_form_consistency", "results": summary_rows}
