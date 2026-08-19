"""CLI entry point: python -m eval.run [phase0|tts|all|report|download] [--test NAME]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from eval.config import load_config
from eval.utils import setup_logging


def _run_phase0(config):
    from eval.phase0.discovery import run
    return run(config)


def _run_tts_all(config):
    from eval.tts import (
        naturalness, intelligibility, prosody, signal_quality,
        latency, concurrency, edge_cases, long_form,
    )
    results = []
    tests = [
        ("2.1", naturalness),
        ("2.2", intelligibility),
        ("2.3", prosody),
        ("2.4", signal_quality),
        ("2.5", latency),
        ("2.6", concurrency),
        ("2.7", edge_cases),
        ("2.8", long_form),
    ]
    for tid, mod in tests:
        logging.getLogger("eval.run").info("--- Running TTS Test %s ---", tid)
        try:
            r = mod.run(config)
            results.append(r)
        except Exception as e:
            logging.getLogger("eval.run").error("TTS Test %s failed: %s", tid, e)
            results.append({"test": tid, "error": str(e)})
    return results


def _run_single_tts(name: str, config):
    mod_map = {
        "naturalness": "eval.tts.naturalness",
        "intelligibility": "eval.tts.intelligibility",
        "prosody": "eval.tts.prosody",
        "signal_quality": "eval.tts.signal_quality",
        "latency": "eval.tts.latency",
        "concurrency": "eval.tts.concurrency",
        "edge_cases": "eval.tts.edge_cases",
        "long_form": "eval.tts.long_form",
    }
    key = name.replace("tts.", "").replace("2.", "").strip()
    num_map = {
        "1": "naturalness", "2": "intelligibility", "3": "prosody",
        "4": "signal_quality", "5": "latency", "6": "concurrency",
        "7": "edge_cases", "8": "long_form",
    }
    if key in num_map:
        key = num_map[key]
    if key not in mod_map:
        print(f"Unknown TTS test: {name}. Available: {list(mod_map)}")
        sys.exit(1)
    import importlib
    mod = importlib.import_module(mod_map[key])
    return mod.run(config)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m eval.run",
        description="KugelAudio TTS Evaluation Harness",
    )
    parser.add_argument(
        "command",
        choices=["phase0", "tts", "all", "report", "download"],
        help="What to run",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Run a single test, e.g. --test accuracy or --test naturalness",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: eval/config.yaml)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and datasets but make no API calls",
    )

    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    log = logging.getLogger("eval.run")

    config = load_config(args.config)
    if args.results_dir:
        config.evaluation.results_dir = args.results_dir

    if args.dry_run:
        log.info("[DRY RUN] Config loaded. Command=%s test=%s", args.command, args.test)
        log.info("  Provider : %s", config.tts.provider)
        if config.tts.kugel:
            log.info("  TTS REST : %s", config.tts.kugel.rest_endpoint)
            log.info("  TTS WS   : %s", config.tts.kugel.ws_endpoint)
            log.info("  Model    : %s  Voice: %s", config.tts.kugel.model_id, config.tts.kugel.voice_id)
        log.info("  Results  : %s", config.evaluation.results_dir)
        return

    log.info("Starting evaluation. Command=%s", args.command)

    if args.command == "phase0":
        result = _run_phase0(config)
        log.info("Phase 0 complete: %s", result.get("overall_status", "done"))

    elif args.command == "tts":
        if args.test:
            result = _run_single_tts(args.test, config)
            log.info("Done: %s", result.get("name", args.test))
        else:
            results = _run_tts_all(config)
            passed = sum(1 for r in results if "error" not in r)
            log.info("TTS complete: %d/%d passed", passed, len(results))

    elif args.command == "all":
        log.info("=== Phase 0 ===")
        _run_phase0(config)
        log.info("=== TTS Tests ===")
        _run_tts_all(config)
        log.info("=== Generating Report ===")
        from eval.report.generate_report import generate
        report_path = generate(config.evaluation.results_dir)
        log.info("Report: %s", report_path)

    elif args.command == "download":
        from eval.data.download_datasets import download_all
        log.info("Pre-downloading all evaluation datasets...")
        results = download_all()
        ok = sum(1 for v in results.values() if v is not None)
        log.info("Downloaded %d/%d datasets", ok, len(results))
        for name, ds in results.items():
            status = f"{len(ds)} examples" if ds is not None else "FAILED"
            log.info("  %-20s %s", name, status)

    elif args.command == "report":
        from eval.report.generate_report import generate
        report_path = generate(config.evaluation.results_dir)
        print(f"Report written to: {report_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
