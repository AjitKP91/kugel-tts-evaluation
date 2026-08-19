from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf

logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

NORMALIZE_FOR_WER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])

# Same as above but without RemoveEmptyStrings — use for aggregate (list) calls
# so that empty refs/hyps stay in the list and lengths never diverge.
NORMALIZE_FOR_WER_AGG = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace(",", "")
        .replace(".", "")
        .replace("?", "")
        .replace("!", "")
        .strip()
    )


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    return audio, sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr)


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.duration


# ---------------------------------------------------------------------------
# JSONL persistence (idempotent resume support)
# ---------------------------------------------------------------------------

def write_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_completed_ids(path: str | Path, id_field: str = "id") -> set:
    return {r[id_field] for r in read_jsonl(path) if id_field in r}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_percentiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> tuple[float, float, float]:
    arr = np.array(values)
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(np.mean(arr)), lower, upper


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_with_backoff(
    max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 30.0,
    reraise_if=None,
):
    """Exponential-backoff retry decorator.

    reraise_if: optional callable(exc) -> bool; if it returns True the
    exception is re-raised immediately without retrying (permanent failures).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if reraise_if is not None and reraise_if(e):
                        raise
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence verbose HTTP request logs from httpx/httpcore; keep datasets progress bars
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Result summary helpers
# ---------------------------------------------------------------------------

def save_summary_csv(path: str | Path, rows: list[dict]) -> None:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Summary saved to %s", path)


# ---------------------------------------------------------------------------
# Dataset loader — download, load into memory, delete cache
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def load_dataset_tmp(path: str, split: str, name=None, limit: int | None = None, **kwargs):
    """Download a HF dataset to a temp dir, convert to an in-memory list, delete cache.

    Usage:
        with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean") as examples:
            for ex in examples: ...
    """
    import itertools
    import os
    import shutil
    import tempfile

    from datasets import load_dataset

    cache_dir = tempfile.mkdtemp(prefix="eval_ds_", dir="/tmp")
    try:
        kw: dict = {"split": split, "token": os.environ.get("HF_TOKEN") or True, "cache_dir": cache_dir, **kwargs}
        if name is not None:
            kw["name"] = name
        ds = load_dataset(path, **kw)
        data = list(itertools.islice(ds, limit)) if limit is not None else list(ds)
        del ds
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    yield data
