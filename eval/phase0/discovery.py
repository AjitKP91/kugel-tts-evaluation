"""Phase 0 — Discovery & Infrastructure.

Verify connectivity, confirm response schema, determine supported audio
formats, and establish baseline numbers for all subsequent TTS tests.

KugelAudio is TTS-only (no STT) and REST + WebSocket only (no gRPC), so this
phase probes only the TTS surface: HTTP batch, WebSocket streaming, parameter
sensitivity, smoke tests, and cold-start latency.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.config import Config, load_config
from eval.tts.client_factory import build_tts_client
from eval.utils import write_jsonl, setup_logging

logger = logging.getLogger("eval.phase0")

SMOKE_TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank "
    "on a warm sunny afternoon."
)


# ------------------------------------------------------------------
# 0.1  Connectivity & Auth
# ------------------------------------------------------------------

def check_connectivity(tts_client, out_dir: Path) -> dict:
    logger.info("=== 0.1 Connectivity & Auth ===")
    results: dict = {"rest_tts": None, "ws_tts": None}

    # REST batch
    try:
        t0 = time.perf_counter()
        resp = tts_client.synthesize_batch("Hello, testing connectivity.")
        latency = time.perf_counter() - t0
        results["rest_tts"] = {
            "status": "ok",
            "latency_s": latency,
            "audio_len_bytes": len(resp["audio_bytes"]),
        }
        logger.info("REST TTS: OK (%.2fs)", latency)
    except Exception as e:
        results["rest_tts"] = {"status": "error", "error": str(e)}
        logger.error("REST TTS failed: %s", e)

    # WebSocket streaming
    try:
        t0 = time.perf_counter()
        resp = tts_client.synthesize_stream("Hello, testing connectivity.")
        latency = time.perf_counter() - t0
        results["ws_tts"] = {
            "status": "ok",
            "latency_s": latency,
            "ttfb_s": resp.get("ttfb"),
            "n_chunks": resp.get("n_chunks"),
        }
        logger.info("WebSocket TTS: OK (%.2fs, ttfb=%.2fs)", latency, resp.get("ttfb") or -1)
    except Exception as e:
        results["ws_tts"] = {"status": "error", "error": str(e)}
        logger.error("WebSocket TTS failed: %s", e)

    # Network baseline (HEAD to the REST host)
    try:
        import requests as req
        t0 = time.perf_counter()
        req.head(tts_client.cfg.rest_endpoint, timeout=10)
        results["network_rtt_s"] = time.perf_counter() - t0
    except Exception:
        results["network_rtt_s"] = None

    return results


# ------------------------------------------------------------------
# 0.3  TTS Schema Discovery
# ------------------------------------------------------------------

def discover_tts_schema(tts_client, out_dir: Path) -> dict:
    logger.info("=== 0.3 TTS Schema Discovery ===")

    text = "Hello, this is a test."

    # Batch (REST)
    batch_result = tts_client.synthesize_batch(text)
    audio_bytes = batch_result["audio_bytes"]

    # Streaming (WebSocket)
    stream_result = tts_client.synthesize_stream(text)

    # Save sample audio
    sample_path = out_dir / "tts_sample.wav"
    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(str(sample_path), audio_arr, tts_client.sample_rate)

    schema = {
        "batch_audio_bytes": len(audio_bytes),
        "batch_audio_duration_s": batch_result["audio_duration"],
        "batch_latency_s": batch_result["elapsed_s"],
        "sample_rate": tts_client.sample_rate,
        "stream_ttfb": stream_result.get("ttfb"),
        "stream_n_chunks": stream_result.get("n_chunks"),
        "stream_audio_duration_s": stream_result["audio_duration"],
        "stream_total_latency_s": stream_result["elapsed_s"],
        "sample_saved": str(sample_path),
    }

    logger.info("TTS schema: %s", json.dumps(schema, indent=2, default=str))
    return schema


# ------------------------------------------------------------------
# 0.4  Parameter Exploration
# ------------------------------------------------------------------

def explore_parameters(tts_client, out_dir: Path) -> dict:
    logger.info("=== 0.4 Parameter Exploration ===")
    results: dict = {"tts_sample_rates": {}, "tts_markup": None}

    # Sample-rate acceptance — Kugel documents 8k/16k/22.05k/24k/44.1k.
    base_sr = tts_client.cfg.sample_rate
    for test_sr in [8000, 16000, 22050, 24000, 44100]:
        try:
            tts_client.cfg.sample_rate = test_sr
            resp = tts_client.synthesize_batch("Sample rate probe.")
            results["tts_sample_rates"][test_sr] = "ok" if resp["audio_duration"] > 0 else "no_audio"
        except Exception as e:
            results["tts_sample_rates"][test_sr] = f"error: {e}"
        finally:
            tts_client.cfg.sample_rate = base_sr

    # Markup handling — Kugel has no documented SSML; this probes whether tag
    # characters are read literally or rejected (plain-text robustness probe).
    try:
        markup_text = '<speak>Hello <break time="500ms"/> world.</speak>'
        resp = tts_client.synthesize_batch(markup_text)
        results["tts_markup"] = "accepted" if resp["audio_duration"] > 0 else "no_audio"
    except Exception as e:
        results["tts_markup"] = f"error: {e}"

    logger.info("Parameters: %s", json.dumps(results, indent=2))
    return results


# ------------------------------------------------------------------
# 0.5  Smoke Tests
# ------------------------------------------------------------------

def smoke_tests(tts_client, out_dir: Path) -> dict:
    logger.info("=== 0.5 Smoke Tests ===")
    results = {}

    # TTS batch (REST)
    try:
        resp = tts_client.synthesize_batch_rest(SMOKE_TEXT)
        dur = resp["audio_duration"]
        results["tts_rest"] = {
            "status": "pass" if dur > 1.0 else "fail",
            "audio_duration_s": dur,
        }
    except Exception as e:
        results["tts_rest"] = {"status": "fail", "error": str(e)}

    # TTS WebSocket streaming
    try:
        resp = tts_client.synthesize_stream(SMOKE_TEXT)
        results["tts_ws_stream"] = {
            "status": "pass" if resp.get("ttfb") and resp["ttfb"] < 5.0 else "fail",
            "ttfb_s": resp.get("ttfb"),
        }
    except Exception as e:
        results["tts_ws_stream"] = {"status": "fail", "error": str(e)}

    for name, r in results.items():
        logger.info("Smoke %s: %s", name, r["status"])

    return results


# ------------------------------------------------------------------
# 0.6  Cold-Start vs Warm Latency
# ------------------------------------------------------------------

def cold_start_test(tts_client, out_dir: Path) -> dict:
    logger.info("=== 0.6 Cold-Start Test ===")
    logger.info(
        "NOTE: For accurate cold-start measurement, run this immediately after "
        "a fresh deployment or after >30 min idle."
    )

    results: dict = {}

    text = "This is a cold start test sentence for the TTS model."
    t0 = time.perf_counter()
    tts_client.synthesize_batch(text)
    t_first = time.perf_counter() - t0

    warm_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        tts_client.synthesize_batch(text)
        warm_times.append(time.perf_counter() - t0)

    t_warm = np.mean(warm_times)
    results["tts"] = {
        "t_first_s": t_first,
        "t_warm_mean_s": float(t_warm),
        "cold_ratio": t_first / t_warm if t_warm > 0 else None,
    }

    for model, r in results.items():
        status = "PASS" if (r["cold_ratio"] or 999) < 5 else "FLAG"
        logger.info(
            "%s cold-start: first=%.2fs, warm=%.2fs, ratio=%.1f (%s)",
            model.upper(), r["t_first_s"], r["t_warm_mean_s"],
            r["cold_ratio"] or -1, status,
        )

    return results


# ------------------------------------------------------------------
# Run all Phase 0
# ------------------------------------------------------------------

def _run_subphase(name: str, fn, out_dir: Path, results: dict, *args, **kwargs) -> None:
    """Run a sub-phase and record its result (or full exception) in results[name].

    On success: results[name] is whatever fn returned.
    On failure: results[name] = {"phase", "error", "error_type", "traceback"}.
    The full traceback also lands in run.log via logger.exception.

    Persists results/phase0/discovery.json after each sub-phase so a crash
    midway leaves everything up to that point on disk.
    """
    import traceback as _tb
    try:
        results[name] = fn(*args, **kwargs)
    except Exception as e:
        tb_str = _tb.format_exc()
        logger.exception("Sub-phase %s FAILED: %s", name, e)
        results[name] = {
            "phase": name,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb_str,
        }
    _write_discovery(out_dir, results)


def _write_discovery(out_dir: Path, results: dict) -> None:
    """Atomic-ish write of discovery.json — write to .tmp then rename."""
    output_path = out_dir / "discovery.json"
    tmp_path = out_dir / "discovery.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp_path.replace(output_path)


def run(config: Config | None = None) -> dict:
    if config is None:
        config = load_config()

    setup_logging(config.evaluation.log_level)
    out_dir = Path(config.evaluation.results_dir) / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Client construction can itself fail (missing API key, invalid config).
    # Wrap it so we still write a usable discovery.json instead of crashing.
    results: dict = {}
    try:
        tts_client = build_tts_client(config)
    except Exception as e:
        logger.exception("Failed to construct TTS client: %s", e)
        results["tts_client_init"] = {
            "phase": "tts_client_init",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        _write_discovery(out_dir, results)
        logger.error(
            "Phase 0 aborted: could not construct TTS client. "
            "Partial discovery.json written to %s", out_dir / "discovery.json",
        )
        return results

    # Each sub-phase is isolated — a failure records its error and moves on.
    _run_subphase("connectivity",  check_connectivity,  out_dir, results, tts_client, out_dir)
    _run_subphase("tts_schema",    discover_tts_schema, out_dir, results, tts_client, out_dir)
    _run_subphase("parameters",    explore_parameters,  out_dir, results, tts_client, out_dir)
    _run_subphase("smoke_tests",   smoke_tests,         out_dir, results, tts_client, out_dir)
    _run_subphase("cold_start",    cold_start_test,     out_dir, results, tts_client, out_dir)

    logger.info("Phase 0 results written to %s", out_dir / "discovery.json")

    passed, failed = [], []
    for name, r in results.items():
        if isinstance(r, dict) and r.get("error"):
            failed.append(name)
        else:
            passed.append(name)
    logger.info("Phase 0 summary: %d passed, %d failed", len(passed), len(failed))
    if failed:
        logger.info("  Failed sub-phases: %s", ", ".join(failed))
        logger.info("  See discovery.json (error/traceback fields) and run.log for details.")

    for wav in out_dir.glob("*.wav"):
        wav.unlink(missing_ok=True)
    logger.info("Phase 0 temporary WAV files deleted.")

    return results


if __name__ == "__main__":
    run()
