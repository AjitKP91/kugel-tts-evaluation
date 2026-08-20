"""Recompute Test 2.8 speaker similarity using ECAPA-TDNN (SpeechBrain).

The original run had speechbrain unavailable, leaving mean_speaker_sim null
for all 5 passages. This script re-synthesizes each passage's paragraphs,
computes ECAPA-TDNN embeddings, and patches the JSONL + summary CSV.

Usage (on the VM, from the repo root):
    pip install speechbrain   # if not already installed
    python scripts/recompute_speaker_similarity.py
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

# Make `import eval` work when run as `python scripts/recompute_speaker_similarity.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

# Compatibility patch: SpeechBrain calls torch.amp.custom_fwd(device_type='cuda')
# which requires PyTorch >= 2.4. On older versions the API lives under
# torch.cuda.amp and doesn't accept a device_type argument.
import torch
if not hasattr(torch.amp, "custom_fwd"):
    def _custom_fwd_compat(fn=None, *, device_type=None, **kwargs):
        # device_type is new-API only — drop it; forward everything else to old API
        if fn is not None:
            return torch.cuda.amp.custom_fwd(fn, **kwargs)   # type: ignore[attr-defined]
        if kwargs:
            return torch.cuda.amp.custom_fwd(**kwargs)        # type: ignore[attr-defined]
        return torch.cuda.amp.custom_fwd                      # type: ignore[attr-defined]
    def _custom_bwd_compat(fn=None, **kwargs):
        if fn is not None:
            return torch.cuda.amp.custom_bwd(fn, **kwargs)   # type: ignore[attr-defined]
        return torch.cuda.amp.custom_bwd                      # type: ignore[attr-defined]
    torch.amp.custom_fwd = _custom_fwd_compat                 # type: ignore[attr-defined]
    torch.amp.custom_bwd = _custom_bwd_compat                 # type: ignore[attr-defined]

from eval.config import load_config
from eval.data.tts_test_sets import get_long_form_passages
from eval.tts.client_factory import build_tts_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

JSONL   = Path("results/tts/long_form/calls.jsonl")
SUMMARY = Path("results/tts/long_form/summary.csv")


def load_encoder():
    """Load ECAPA-TDNN — handles both speechbrain <1.0 and >=1.0 APIs."""
    try:
        # speechbrain >= 1.0
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        # speechbrain < 1.0
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
    )


def embed(encoder, wav_path: str) -> np.ndarray:
    import torch
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.tensor(wav).unsqueeze(0)
    emb = encoder.encode_batch(wav_t).squeeze().cpu().detach().numpy()
    return emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def pairwise_mean_sim(embeddings: list[np.ndarray]) -> tuple[float, float]:
    sims = [
        cosine_sim(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    return float(np.mean(sims)), float(np.std(sims))


def main():
    logger.info("Loading ECAPA-TDNN encoder...")
    encoder = load_encoder()
    logger.info("Encoder ready.")

    config = load_config()
    tts_client = build_tts_client(config)

    passages = get_long_form_passages()

    # Load existing records keyed by passage_id
    records: dict[int, dict] = {}
    for line in JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            records[r["passage_id"]] = r

    for pi, passage in enumerate(passages):
        rec = records.get(pi)
        if rec and rec.get("mean_speaker_sim") is not None:
            logger.info("Passage %d already has speaker_sim=%.4f — skipping", pi, rec["mean_speaker_sim"])
            continue

        logger.info("Passage %d: %s (%d paragraphs)", pi, passage["title"], len(passage["paragraphs"]))
        embeddings = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for pj, para in enumerate(passage["paragraphs"]):
                wav_path = Path(tmpdir) / f"para_{pj:02d}.wav"
                try:
                    tts_client.save_synthesis(para, str(wav_path))
                    emb = embed(encoder, str(wav_path))
                    embeddings.append(emb)
                    logger.info("  Para %d: embedding shape %s", pj, emb.shape)
                except Exception as e:
                    logger.warning("  Para %d failed: %s", pj, e)

        if len(embeddings) < 2:
            logger.warning("  Not enough embeddings for passage %d — skipping", pi)
            continue

        mean_sim, sim_std = pairwise_mean_sim(embeddings)
        logger.info("  mean_speaker_sim=%.4f  std=%.4f", mean_sim, sim_std)

        if rec:
            rec["mean_speaker_sim"] = round(mean_sim, 4)
            rec["speaker_sim_std"]  = round(sim_std, 4)
        else:
            records[pi] = {
                "passage_id": pi,
                "title": passage["title"],
                "n_paragraphs": len(embeddings),
                "mean_speaker_sim": round(mean_sim, 4),
                "speaker_sim_std": round(sim_std, 4),
                "f0_drift_hz": None,
                "wpm_drift": None,
            }

    # Rewrite JSONL
    with JSONL.open("w") as f:
        for r in records.values():
            f.write(json.dumps(r, default=str) + "\n")

    # Rewrite summary CSV
    import pandas as pd
    df = pd.DataFrame(list(records.values()))
    df.to_csv(SUMMARY, index=False)
    logger.info("Written %s", SUMMARY)

    # Print summary
    sims = [r["mean_speaker_sim"] for r in records.values() if r.get("mean_speaker_sim") is not None]
    if sims:
        logger.info("Speaker similarity across passages: mean=%.4f  min=%.4f  max=%.4f",
                    np.mean(sims), np.min(sims), np.max(sims))


if __name__ == "__main__":
    main()
