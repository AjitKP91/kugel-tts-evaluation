"""Download all evaluation datasets from HuggingFace and other sources."""
from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger("eval.data")

DATASETS = {
    # ---- TTS reference datasets ----
    # LJSpeech is the only external dataset the TTS suite uses (as an optional
    # matched-speaker reference for Test 2.4, which self-skips for Kugel unless
    # a reference set is configured). All in-suite text (Harvard sentences,
    # intelligibility/edge-case/long-form sets) is bundled in eval/data/.
    "ljspeech": {
        "hf_path": "keithito/lj_speech",
        "hf_name": None,
        "split": "train",
        "description": "LJSpeech (single female speaker, TTS reference)",
    },
}


def download_dataset(name: str, cache_dir: str | Path | None = None) -> object:
    info = DATASETS[name]
    logger.info("Downloading %s: %s", name, info["description"])

    kwargs = {
        "path": info["hf_path"],
        "split": info["split"],
        "token": True,
    }
    if info.get("hf_name"):
        kwargs["name"] = info["hf_name"]
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset(**kwargs)
    logger.info("Downloaded %s: %d examples", name, len(ds))
    return ds


def download_all(cache_dir: str | Path | None = None) -> dict:
    results = {}
    for name in DATASETS:
        try:
            results[name] = download_dataset(name, cache_dir)
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            results[name] = None
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "all":
        download_all(args.cache_dir)
    else:
        download_dataset(args.dataset, args.cache_dir)
