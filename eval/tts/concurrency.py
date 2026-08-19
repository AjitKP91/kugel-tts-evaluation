"""Test 2.6 — Throughput & Concurrency."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path

import aiohttp
import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.tts.client_factory import build_tts_client
from eval.utils import compute_percentiles, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.concurrency")

TEST_TEXT = (
    "The artificial intelligence revolution is transforming industries across the globe, "
    "from healthcare and finance to transportation and entertainment."
)

_CALL_TIMEOUT_S = 60
_LEVEL_WALL_TIMEOUT_S = 300  # abandon a concurrency level after 5 minutes total


async def _rest_request(session: aiohttp.ClientSession, url: str, headers: dict, payload: dict) -> dict:
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            data = await resp.read()
            elapsed = time.perf_counter() - start
            return {"elapsed_s": elapsed, "status": resp.status, "audio_len": len(data)}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"elapsed_s": elapsed, "status": -1, "error": str(e)}


async def _run_concurrent_rest(config: Config, n_concurrent: int, n_total: int) -> list[dict]:
    kugel = config.tts.kugel
    headers = {
        "Authorization": f"Bearer {kugel.api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "text": TEST_TEXT,
        "model_id": kugel.model_id,
        "voice_id": kugel.voice_id,
        "cfg_scale": kugel.cfg_scale,
        "temperature": kugel.temperature,
        "speed": kugel.speed,
        "normalize": kugel.normalize,
        "sample_rate": kugel.sample_rate,
    }
    if kugel.language:
        payload["language"] = kugel.language

    sem = asyncio.Semaphore(n_concurrent)
    results = []

    async def bounded_request(session):
        async with sem:
            return await _rest_request(session, kugel.rest_endpoint, headers, payload)

    timeout = aiohttp.ClientTimeout(total=kugel.request_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [bounded_request(session) for _ in range(n_total)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        r if isinstance(r, dict) else {"elapsed_s": 0, "status": -1, "error": str(r)}
        for r in results
    ]


def _run_concurrent_threadpool(tts_client, n_concurrent: int, n_total: int) -> tuple[list[dict], float, bool]:
    """Fan out synchronous batch calls across a thread pool.

    Returns (results, wall_elapsed_s, wall_timeout_hit). wall_timeout_hit=True
    means some futures were abandoned at _LEVEL_WALL_TIMEOUT_S; in that case
    wall_elapsed_s is unreliable as an RPS denominator.
    """
    results: list[dict] = []
    wall_start = time.perf_counter()
    wall_timeout_hit = False

    def _call(_):
        return tts_client.synthesize_batch(TEST_TEXT)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent)
    try:
        futures = [executor.submit(_call, i) for i in range(n_total)]
        done, not_done = concurrent.futures.wait(futures, timeout=_LEVEL_WALL_TIMEOUT_S)

        for f in done:
            try:
                r = f.result(timeout=1)
                results.append({"elapsed_s": r["elapsed_s"], "status": 200, "audio_duration": r["audio_duration"]})
            except Exception as e:
                results.append({"elapsed_s": 0, "status": -1, "error": str(e)})

        if not_done:
            wall_timeout_hit = True
            logger.warning("  threadpool N=%d: %d futures timed out after %ds, moving on",
                           n_concurrent, len(not_done), _LEVEL_WALL_TIMEOUT_S)
            for f in not_done:
                results.append({"elapsed_s": _LEVEL_WALL_TIMEOUT_S, "status": -1, "error": "wall_timeout"})
    finally:
        # wait=False so we don't block on hung threads; cancel_futures drops queued work
        executor.shutdown(wait=False, cancel_futures=True)

    wall_elapsed = time.perf_counter() - wall_start
    return results, wall_elapsed, wall_timeout_hit


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "concurrency"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = build_tts_client(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.6: Throughput & Concurrency ===")

    n_total_per_level = 50
    summary_rows = []

    for n_concurrent in config.evaluation.tts_concurrency_levels:
        logger.info("Concurrency N=%d", n_concurrent)

        # Thread-pool concurrency (synchronous batch client)
        logger.info("  threadpool...")
        grpc_results, grpc_wall, grpc_timeout_hit = _run_concurrent_threadpool(tts_client, n_concurrent, n_total_per_level)
        grpc_latencies = [r["elapsed_s"] for r in grpc_results if r["status"] == 200]
        grpc_errors = sum(1 for r in grpc_results if r["status"] != 200)

        if not grpc_timeout_hit and grpc_wall > 0:
            grpc_rps = len(grpc_latencies) / grpc_wall
            grpc_rps_method = "measured"
        elif grpc_latencies:
            # Wall clock contaminated by hung requests — use Little's Law on successful requests only.
            # Effective concurrency weighted by success rate; mean latency from successful requests.
            effective_conc = (len(grpc_latencies) / len(grpc_results)) * n_concurrent
            grpc_rps = effective_conc / np.mean(grpc_latencies)
            grpc_rps_method = "estimated"
        else:
            grpc_rps = None
            grpc_rps_method = "n/a"

        if grpc_latencies:
            grpc_stats = compute_percentiles(grpc_latencies)
            row = {
                "interface": "threadpool",
                "concurrency": n_concurrent,
                "n_calls": len(grpc_results),
                "n_success": len(grpc_latencies),
                "n_errors": grpc_errors,
                "error_rate": round(grpc_errors / len(grpc_results), 4),
                "rps": round(grpc_rps, 2) if grpc_rps is not None else None,
                "rps_method": grpc_rps_method,
                **{f"latency_{k}": round(v, 3) for k, v in grpc_stats.items()},
            }
            summary_rows.append(row)
            write_jsonl(jsonl_path, row)

        # REST concurrency
        logger.info("  REST...")
        try:
            rest_results = asyncio.run(
                _run_concurrent_rest(config, n_concurrent, n_total_per_level)
            )
            rest_latencies = [r["elapsed_s"] for r in rest_results if r.get("status") == 200]
            rest_errors = sum(1 for r in rest_results if r.get("status") != 200)

            if rest_latencies:
                rest_stats = compute_percentiles(rest_latencies)
                rest_total_time = sum(rest_latencies) / n_concurrent if n_concurrent > 0 else sum(rest_latencies)
                row = {
                    "interface": "rest",
                    "concurrency": n_concurrent,
                    "n_calls": len(rest_results),
                    "n_success": len(rest_latencies),
                    "n_errors": rest_errors,
                    "error_rate": round(rest_errors / len(rest_results), 4),
                    "rps": round(len(rest_latencies) / rest_total_time, 2) if rest_total_time > 0 else 0,
                    **{f"latency_{k}": round(v, 3) for k, v in rest_stats.items()},
                }
                summary_rows.append(row)
                write_jsonl(jsonl_path, row)
        except Exception as e:
            logger.warning("REST concurrency test failed: %s", e)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    for row in summary_rows:
        logger.info(
            "%s N=%d: P50=%.0fms P99=%.0fms RPS=%.1f errors=%d",
            row["interface"], row["concurrency"],
            row.get("latency_p50", 0) * 1000, row.get("latency_p99", 0) * 1000,
            row.get("rps", 0), row.get("n_errors", 0),
        )

    return {"test": "2.6", "name": "concurrency", "results": summary_rows}
