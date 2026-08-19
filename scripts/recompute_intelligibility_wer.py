"""Recompute Test 2.2 WER using Whisper's EnglishTextNormalizer on both sides.

The original run compared spelled-out-word references ("one thousand two
hundred and thirty-four dollars and fifty-six cents") against Whisper
hypotheses in digit form ("$1,234.56"), inflating numbers WER to ~55%.
This script applies the same normalizer to both sides so "one thousand ..."
and "$1,234.56" collapse to the same token sequence before WER is measured.

Usage (on the VM, from the repo root):
    python scripts/recompute_intelligibility_wer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import jiwer
from whisper.normalizers import EnglishTextNormalizer

JSONL = Path("results/tts/intelligibility/calls.jsonl")
SUMMARY = Path("results/tts/intelligibility/summary.csv")

normalizer = EnglishTextNormalizer()

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def whisper_normalize(text: str) -> str:
    return normalizer(text)


def recompute():
    rows = []
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Group by category
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    summary_rows = []
    for cat, items in by_cat.items():
        refs_raw  = [r["original_text"]      for r in items]
        hyps_raw  = [r["whisper_transcript"]  for r in items]

        refs_norm = [whisper_normalize(r) for r in refs_raw]
        hyps_norm = [whisper_normalize(h) for h in hyps_raw]

        agg_wer = jiwer.wer(refs_norm, hyps_norm,
                            reference_transform=transform,
                            hypothesis_transform=transform)
        agg_cer = jiwer.cer(refs_norm, hyps_norm,
                            reference_transform=transform,
                            hypothesis_transform=transform)

        # Also store per-sentence normalised WER back in rows
        for r, ref_n, hyp_n in zip(items, refs_norm, hyps_norm):
            r["wer_norm"] = round(jiwer.wer(ref_n, hyp_n,
                                            reference_transform=transform,
                                            hypothesis_transform=transform), 4)
            r["cer_norm"] = round(jiwer.cer(ref_n, hyp_n,
                                            reference_transform=transform,
                                            hypothesis_transform=transform), 4)

        n = len(items)
        print(f"{cat:20s}  n={n}  WER(orig)={sum(r['wer'] for r in items)/n:.1%}"
              f"  WER(norm)={agg_wer:.1%}  CER(norm)={agg_cer:.1%}")

        summary_rows.append({
            "category": cat,
            "n_sentences": n,
            "round_trip_wer": round(agg_wer, 4),
            "round_trip_cer": round(agg_cer, 4),
        })

    # Rewrite JSONL with normalised per-sentence scores added
    with JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    # Rewrite summary CSV
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    df.to_csv(SUMMARY, index=False)
    print(f"\nSummary written to {SUMMARY}")


if __name__ == "__main__":
    recompute()
