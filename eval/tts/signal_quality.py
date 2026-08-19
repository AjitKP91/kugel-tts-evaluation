"""Test 2.4 — Audio Signal Quality (MCD / PESQ / STOI)."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from eval.config import Config
from eval.tts.client_factory import build_tts_client
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.signal_quality")

TARGET_SR = 16000


def compute_mcd(ref_path: str, syn_path: str, n_mfcc: int = 13) -> float:
    """Mel Cepstral Distortion via librosa + DTW."""
    from dtw import dtw

    ref, sr = librosa.load(ref_path, sr=22050)
    syn, _ = librosa.load(syn_path, sr=22050)
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc).T
    syn_mfcc = librosa.feature.mfcc(y=syn, sr=sr, n_mfcc=n_mfcc).T
    alignment = dtw(ref_mfcc, syn_mfcc, dist_method="euclidean")
    return (10.0 * np.sqrt(2) / np.log(10)) * alignment.distance / len(ref_mfcc)


def compute_pesq_score(ref_path: str, syn_path: str) -> float | None:
    try:
        from pesq import pesq
        ref, sr_ref = sf.read(ref_path)
        syn, sr_syn = sf.read(syn_path)
        # Resample to 16kHz
        if sr_ref != TARGET_SR:
            ref = librosa.resample(ref.astype(np.float32), orig_sr=sr_ref, target_sr=TARGET_SR)
        if sr_syn != TARGET_SR:
            syn = librosa.resample(syn.astype(np.float32), orig_sr=sr_syn, target_sr=TARGET_SR)
        min_len = min(len(ref), len(syn))
        return float(pesq(TARGET_SR, ref[:min_len], syn[:min_len], "wb"))
    except Exception as e:
        logger.warning("PESQ failed: %s", e)
        return None


def compute_stoi_score(ref_path: str, syn_path: str) -> float | None:
    try:
        from pystoi import stoi
        ref, sr_ref = sf.read(ref_path)
        syn, sr_syn = sf.read(syn_path)
        if sr_ref != TARGET_SR:
            ref = librosa.resample(ref.astype(np.float32), orig_sr=sr_ref, target_sr=TARGET_SR)
        if sr_syn != TARGET_SR:
            syn = librosa.resample(syn.astype(np.float32), orig_sr=sr_syn, target_sr=TARGET_SR)
        min_len = min(len(ref), len(syn))
        return float(stoi(ref[:min_len], syn[:min_len], TARGET_SR, extended=True))
    except Exception as e:
        logger.warning("STOI failed: %s", e)
        return None


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "signal_quality"
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.4: Audio Signal Quality ===")

    # PESQ/STOI/MCD require a *same-speaker* reference: synthesized audio vs a
    # real recording of the same voice. A stock Kugel library voice has no such
    # ground truth, so the test needs a matched-speaker reference set built by
    # cloning a single-speaker corpus (LJSpeech) into a Kugel voice. Run
    # scripts/clone_reference_voice.py to produce it, then set
    # tts.kugel.reference_set_dir + reference_voice_id. Without it, skip.
    #
    # NOTE: when enabled, this measures how faithfully Kugel reproduces the
    # cloned speaker (clone fidelity) — not the quality of a stock voice.
    provider = (getattr(config.tts, "provider", None) or "kugel").lower()
    ref_dir = None
    ref_voice_id = None
    if provider == "kugel":
        kugel = config.tts.kugel
        ref_dir = getattr(kugel, "reference_set_dir", None) if kugel else None
        ref_voice_id = getattr(kugel, "reference_voice_id", None) if kugel else None
        if not ref_dir:
            logger.warning(
                "Test 2.4 skipped for provider=kugel — no matched-speaker "
                "reference set configured. Build one with "
                "`python scripts/clone_reference_voice.py`, then set "
                "tts.kugel.reference_set_dir + reference_voice_id. "
                "Rely on Tests 2.1 and 2.2 for quality in the meantime."
            )
            return {
                "test": "2.4",
                "name": "signal_quality",
                "results": {
                    "skipped": True,
                    "reason": "kugel_no_reference_set",
                    "note": (
                        "PESQ/STOI/MCD require a same-speaker reference. Run "
                        "scripts/clone_reference_voice.py to clone LJSpeech into "
                        "a Kugel voice and enable this test."
                    ),
                },
            }

    tts_client = build_tts_client(config)

    # Assemble the (text, reference_wav) pairs.
    ref_items: list[dict] = []
    if provider == "kugel":
        # Cloned matched-speaker reference produced by clone_reference_voice.py.
        ref_path = Path(ref_dir)
        manifest_path = ref_path / "manifest.json"
        if not manifest_path.exists():
            logger.warning("reference_set_dir %s has no manifest.json — skipping.", ref_dir)
            return {"test": "2.4", "name": "signal_quality",
                    "results": {"skipped": True, "reason": "missing_manifest"}}
        import json
        manifest = json.loads(manifest_path.read_text())
        ref_voice_id = ref_voice_id or manifest.get("voice_id")
        for m in manifest.get("items", []):
            wav = ref_path / m["wav"]
            if wav.exists():
                ref_items.append({"text": m["text"], "ref_wav": str(wav)})
        logger.info("Loaded %d matched-speaker reference clips; cloned voice_id=%s",
                    len(ref_items), ref_voice_id)
    else:
        # Riva/other path: use LJSpeech directly (original behaviour).
        try:
            from eval.utils import load_dataset_tmp
            with load_dataset_tmp("keithito/lj_speech", "train", limit=50) as lj_items:
                for i, item in enumerate(lj_items):
                    text = item.get("normalized_text") or item.get("text", "")
                    ref_audio = np.array(item["audio"]["array"], dtype=np.float32)
                    ref_sr = item["audio"]["sampling_rate"]
                    p = results_dir / f"ref_{i:04d}.wav"
                    sf.write(str(p), ref_audio, ref_sr)
                    ref_items.append({"text": text, "ref_wav": str(p), "_tmp": True})
        except Exception as e:
            logger.warning("LJSpeech not available: %s. Skipping Test 2.4.", e)

    if not ref_items:
        logger.warning("No reference clips available — skipping Test 2.4.")
        return {"test": "2.4", "name": "signal_quality", "results": {"skipped": True}}

    mcd_values = []
    pesq_values = []
    stoi_values = []

    for i, item in enumerate(tqdm(ref_items, desc="Signal quality")):
        text = item["text"]
        ref_path_i = item["ref_wav"]
        syn_path = results_dir / f"syn_{i:04d}.wav"

        try:
            if provider == "kugel":
                tts_client.save_synthesis(text, str(syn_path), voice_id=ref_voice_id)
            else:
                tts_client.save_synthesis(text, str(syn_path))

            mcd = compute_mcd(ref_path_i, str(syn_path))
            pesq_val = compute_pesq_score(ref_path_i, str(syn_path))
            stoi_val = compute_stoi_score(ref_path_i, str(syn_path))

            mcd_values.append(mcd)
            if pesq_val is not None:
                pesq_values.append(pesq_val)
            if stoi_val is not None:
                stoi_values.append(stoi_val)

            write_jsonl(jsonl_path, {
                "id": f"sigq_{i}",
                "text": text,
                "mcd": round(mcd, 2),
                "pesq": round(pesq_val, 3) if pesq_val else None,
                "stoi": round(stoi_val, 4) if stoi_val else None,
            })
        except Exception as e:
            logger.warning("Failed signal quality %d: %s", i, e)
        finally:
            syn_path.unlink(missing_ok=True)
            if item.get("_tmp"):
                Path(ref_path_i).unlink(missing_ok=True)

    summary = {
        "reference_type": (
            "cloned matched-speaker (LJSpeech → Kugel voice); metrics reflect "
            "clone fidelity, not stock-voice quality"
            if provider == "kugel"
            else "LJSpeech reference (may differ from synthesized speaker)"
        ),
        "reference_voice_id": ref_voice_id,
        "mcd_mean": round(np.mean(mcd_values), 2) if mcd_values else None,
        "pesq_mean": round(np.mean(pesq_values), 3) if pesq_values else None,
        "stoi_mean": round(np.mean(stoi_values), 4) if stoi_values else None,
        "n_samples": len(mcd_values),
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    if mcd_values:
        logger.info("MCD: %.1f dB, PESQ: %.2f, STOI: %.3f",
            np.mean(mcd_values),
            np.mean(pesq_values) if pesq_values else 0,
            np.mean(stoi_values) if stoi_values else 0,
        )

    return {"test": "2.4", "name": "signal_quality", "results": summary}
