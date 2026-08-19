"""Bootstrap the Test 2.4 matched-speaker reference for KugelAudio.

Test 2.4 (MCD / PESQ / STOI) needs a *same-speaker* comparison: synthesized
audio vs a real recording of the same voice saying the same words. A stock
Kugel library voice has no ground-truth human recording, so the test otherwise
self-skips.

This script closes that gap by cloning a single-speaker corpus (LJSpeech) into
a Kugel voice, then saving the corpus's ground-truth WAVs + transcripts as the
reference set. Test 2.4 then synthesizes those transcripts with the cloned
voice and compares against the real recordings.

What it measures afterwards: how faithfully Kugel *reproduces the cloned
speaker* — clone fidelity — not the quality of a stock library voice. That is
the only speaker-matched way to run these metrics; see docs/kugel for the
caveats.

Usage (needs KUGELAUDIO_API_KEY and HF access to keithito/lj_speech):
    python scripts/clone_reference_voice.py \
        --n-clone 6 --n-reference 30 --out eval/data/kugel_reference

Then set in eval/config.yaml (the script prints the exact values):
    tts.kugel.reference_set_dir:  <out>
    tts.kugel.reference_voice_id: <returned voice_id>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.config import load_config
from eval.tts.client_factory import build_tts_client
from eval.utils import load_dataset_tmp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("clone_reference_voice")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone LJSpeech into a Kugel voice for Test 2.4")
    parser.add_argument("--n-clone", type=int, default=6,
                        help="Number of clips uploaded as clone reference audio (10–30 s total is plenty)")
    parser.add_argument("--n-reference", type=int, default=30,
                        help="Number of ground-truth clips saved as the matched-speaker reference set")
    parser.add_argument("--out", type=str, default="eval/data/kugel_reference",
                        help="Directory to write reference WAVs + manifest.json")
    parser.add_argument("--voice-name", type=str, default="ljspeech-clone")
    parser.add_argument("--sex", type=str, default="female")
    args = parser.parse_args()

    config = load_config()
    if (config.tts.provider or "").lower() != "kugel":
        raise SystemExit(f"tts.provider is {config.tts.provider!r}, expected 'kugel'")
    client = build_tts_client(config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = out_dir / "_clone_input"
    clone_dir.mkdir(exist_ok=True)

    total = args.n_clone + args.n_reference
    logger.info("Loading %d LJSpeech clips (%d for cloning, %d for reference)...",
                total, args.n_clone, args.n_reference)
    with load_dataset_tmp("keithito/lj_speech", "train", limit=total) as items:
        rows = list(items)

    if len(rows) < total:
        logger.warning("Only %d clips available; adjusting counts.", len(rows))

    clone_rows = rows[: args.n_clone]
    ref_rows = rows[args.n_clone : total]

    # 1. Write clone-input WAVs and upload them to create the cloned voice.
    clone_paths = []
    for i, item in enumerate(clone_rows):
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]
        p = clone_dir / f"clone_{i:03d}.wav"
        sf.write(str(p), audio, sr)
        clone_paths.append(p)

    logger.info("Uploading %d reference clips to POST /v1/voices ...", len(clone_paths))
    resp = client.clone_voice(
        reference_paths=clone_paths,
        name=args.voice_name,
        sex=args.sex,
        description="LJSpeech single-speaker clone for Test 2.4 matched-speaker reference",
    )
    voice_id = resp.get("voice_id") or resp.get("id")
    if voice_id is None:
        raise SystemExit(f"Clone response missing voice_id: {json.dumps(resp)[:500]}")
    logger.info("Cloned voice_id = %s", voice_id)

    # 2. Save ground-truth reference WAVs + transcripts as the reference set.
    manifest = []
    for i, item in enumerate(ref_rows):
        text = item.get("normalized_text") or item.get("text", "")
        audio = np.array(item["audio"]["array"], dtype=np.float32)
        sr = item["audio"]["sampling_rate"]
        wav_name = f"ref_{i:04d}.wav"
        sf.write(str(out_dir / wav_name), audio, sr)
        manifest.append({"id": f"ref_{i:04d}", "wav": wav_name, "text": text, "sr": sr})

    (out_dir / "manifest.json").write_text(
        json.dumps({"voice_id": voice_id, "items": manifest}, indent=2)
    )
    logger.info("Wrote %d reference WAVs + manifest.json to %s", len(manifest), out_dir)

    # Cleanup clone-input WAVs (not needed after upload).
    for p in clone_paths:
        p.unlink(missing_ok=True)
    try:
        clone_dir.rmdir()
    except OSError:
        pass

    print("\n" + "=" * 60)
    print("Test 2.4 reference ready. Set these in eval/config.yaml:")
    print(f"    tts.kugel.reference_set_dir:  {out_dir}")
    print(f"    tts.kugel.reference_voice_id: {voice_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
