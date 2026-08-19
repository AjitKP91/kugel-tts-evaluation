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

By default it also patches eval/config.yaml in place (comment-preserving) so
Test 2.4 is enabled on the next run. Pass --no-patch-config to only print the
values. Re-running is idempotent: if config.yaml already points at an existing
reference set with a manifest, it exits without re-cloning.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.config import load_config
from eval.tts.client_factory import build_tts_client
from eval.utils import load_dataset_tmp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("clone_reference_voice")


def _patch_config(config_path: Path, ref_dir: str, voice_id) -> bool:
    """Set tts.kugel.reference_set_dir + reference_voice_id in config.yaml,
    preserving comments and layout. Returns True if the file was changed.

    Uses line-level regex substitution (not a YAML round-trip) so the file's
    inline comments and formatting survive. Both keys already exist in the
    shipped config with `null` defaults, so we only rewrite their values.
    """
    text = config_path.read_text()
    voice_repr = str(voice_id) if not isinstance(voice_id, str) else f'"{voice_id}"'

    replacements = {
        "reference_set_dir": f"{ref_dir}",
        "reference_voice_id": voice_repr,
    }
    changed = False
    for key, val in replacements.items():
        # Match the indented "key: <anything>" line, keep leading whitespace
        # and any trailing inline comment.
        pat = re.compile(rf"^(?P<indent>\s*){key}:[^\n#]*(?P<comment>#.*)?$", re.MULTILINE)
        def _sub(m):
            comment = m.group("comment")
            tail = f"  {comment}" if comment else ""
            return f"{m.group('indent')}{key}: {val}{tail}"
        new_text, n = pat.subn(_sub, text)
        if n == 0:
            logger.warning("Could not find '%s:' in %s — leaving it unset.", key, config_path)
        else:
            text = new_text
            changed = True
    if changed:
        config_path.write_text(text)
    return changed


def _already_configured(config) -> bool:
    """True if config already points at a usable reference set (manifest exists)."""
    kugel = config.tts.kugel
    ref_dir = getattr(kugel, "reference_set_dir", None) if kugel else None
    ref_voice = getattr(kugel, "reference_voice_id", None) if kugel else None
    if not ref_dir or not ref_voice:
        return False
    return (Path(ref_dir) / "manifest.json").exists()


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
    parser.add_argument("--config", type=str, default="eval/config.yaml",
                        help="Path to config.yaml to patch")
    parser.add_argument("--no-patch-config", dest="patch_config", action="store_false",
                        help="Only print the values instead of writing them into config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-clone even if a reference set is already configured")
    args = parser.parse_args()

    config = load_config(args.config)
    if (config.tts.provider or "").lower() != "kugel":
        raise SystemExit(f"tts.provider is {config.tts.provider!r}, expected 'kugel'")

    if not args.force and _already_configured(config):
        logger.info(
            "Test 2.4 reference already configured (%s, voice_id=%s) and manifest "
            "present — nothing to do. Use --force to re-clone.",
            config.tts.kugel.reference_set_dir, config.tts.kugel.reference_voice_id,
        )
        return

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

    # 3. Patch config.yaml so Test 2.4 runs on the next eval.
    if args.patch_config:
        cfg_path = Path(args.config)
        if _patch_config(cfg_path, str(out_dir), voice_id):
            logger.info("Patched %s: reference_set_dir=%s reference_voice_id=%s",
                        cfg_path, out_dir, voice_id)
        else:
            logger.warning("Config not patched — set the values below manually.")

    print("\n" + "=" * 60)
    if args.patch_config:
        print("Test 2.4 reference ready and wired into config.yaml:")
    else:
        print("Test 2.4 reference ready. Set these in eval/config.yaml:")
    print(f"    tts.kugel.reference_set_dir:  {out_dir}")
    print(f"    tts.kugel.reference_voice_id: {voice_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
