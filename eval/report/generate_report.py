"""HTML report generator: reads all results/ JSONL + CSVs, applies thresholds, writes report.html."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("eval.report.generate_report")

# Pass/fail thresholds from evaluation plan
THRESHOLDS = {
    "tts": {
        "utmos_mean": {"target": 4.0, "accept": 3.5, "direction": "higher"},
        "round_trip_wer": {"target": 0.03, "accept": 0.05, "direction": "lower"},
        "ttfb_p50_ms": {"target": 200, "accept": 400, "direction": "lower"},
        "speaker_sim": {"target": 0.85, "accept": 0.75, "direction": "higher"},
        "f0_drift_hz": {"target": 15, "accept": 25, "direction": "lower"},
    },
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _read_summary_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        import pandas as pd
        return pd.read_csv(path).to_dict(orient="records")
    except Exception:
        return []


def _try_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # reject NaN
    except (TypeError, ValueError):
        return None


def _status_badge(value, threshold_key: str, category: str) -> str:
    th = THRESHOLDS.get(category, {}).get(threshold_key)
    if th is None or value is None:
        return f'<span class="badge badge-unknown">{value}</span>'
    direction = th["direction"]
    target, accept = th["target"], th["accept"]
    if direction == "lower":
        status = "pass" if value <= target else ("warn" if value <= accept else "fail")
    else:
        status = "pass" if value >= target else ("warn" if value >= accept else "fail")
    color = {"pass": "#28a745", "warn": "#ffc107", "fail": "#dc3545", "unknown": "#6c757d"}[status]
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em">{value:.3f}</span>'


def _table(rows: list[dict], max_rows: int = 50) -> str:
    if not rows:
        return "<p><em>No data</em></p>"
    rows = rows[:max_rows]
    keys = list(rows[0].keys())
    header = "".join(f"<th>{k}</th>" for k in keys)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{row.get(k, '')}</td>" for k in keys)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"""
    <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;width:100%;font-size:0.85em">
      <thead style="background:#f0f0f0"><tr>{header}</tr></thead>
      <tbody>{"".join(body_rows)}</tbody>
    </table>"""


def _section(title: str, content: str, test_id: str = "") -> str:
    anchor = test_id.replace(".", "_")
    return f"""
    <section id="{anchor}" style="margin-bottom:2em">
      <h2 style="border-bottom:2px solid #333;padding-bottom:6px">{title}</h2>
      {content}
    </section>"""


def _collect_tts_results(results_dir: Path) -> dict:
    tts_dir = results_dir / "tts"
    out = {}

    tests = {
        "naturalness": "Test 2.1 — Automated Naturalness",
        "intelligibility": "Test 2.2 — Intelligibility (Round-Trip WER)",
        "prosody": "Test 2.3 — Prosody",
        "signal_quality": "Test 2.4 — Audio Signal Quality",
        "latency": "Test 2.5 — Streaming TTFB & RTF",
        "concurrency": "Test 2.6 — Throughput & Concurrency",
        "edge_cases": "Test 2.7 — Edge Cases & Input Robustness",
        "long_form": "Test 2.8 — Long-Form Voice Consistency",
    }
    for key, label in tests.items():
        test_dir = tts_dir / key
        out[key] = {
            "label": label,
            "summary": _read_summary_csv(test_dir / "summary.csv"),
            "calls": _read_jsonl(test_dir / "calls.jsonl"),
        }
    return out


def _collect_phase0(results_dir: Path) -> dict:
    p0_path = results_dir / "phase0" / "discovery.json"
    if p0_path.exists():
        try:
            return json.loads(p0_path.read_text())
        except Exception:
            pass
    return {}


def _render_phase0(data: dict) -> str:
    if not data:
        return "<p><em>Phase 0 results not found.</em></p>"
    rows = []
    for check, result in data.items():
        if isinstance(result, dict):
            status = result.get("status", "?")
            details = result.get("details", result.get("error", ""))
            color = "#28a745" if status == "pass" else "#dc3545"
            rows.append(f"<tr><td>{check}</td><td style='color:{color};font-weight:bold'>{status}</td><td>{details}</td></tr>")
    if not rows:
        return f"<pre>{json.dumps(data, indent=2, default=str)[:2000]}</pre>"
    header = "<tr><th>Check</th><th>Status</th><th>Details</th></tr>"
    return f"<table border='1' cellpadding='4' style='border-collapse:collapse'><thead style='background:#f0f0f0'>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def _render_tts_section(key: str, data: dict) -> str:
    label = data["label"]
    summary = data["summary"]
    calls = data["calls"]

    content = ""
    if summary:
        content += "<h3>Summary</h3>" + _table(summary)
        if key == "naturalness" and summary:
            row = summary[0] if isinstance(summary[0], dict) else {}
            utmos_raw = row.get("utmos_mean") or row.get("utmos")
            if isinstance(utmos_raw, dict):
                utmos_raw = utmos_raw.get("mean")
            utmos = _try_float(utmos_raw)
            if utmos is not None:
                content += f"<p>UTMOS mean: {_status_badge(utmos, 'utmos_mean', 'tts')}</p>"
        if key == "intelligibility" and summary:
            for row in summary:
                wer = _try_float(row.get("round_trip_wer"))
                if wer is not None:
                    content += f"<p>{row.get('category', '')}: WER = {_status_badge(wer, 'round_trip_wer', 'tts')}</p>"
        if key == "latency" and summary:
            for row in summary:
                ttfb_raw = row.get("ttfb_p50") or (row.get("ttfb") or {}).get("p50") if isinstance(row.get("ttfb"), dict) else row.get("ttfb_p50")
                ttfb = _try_float(ttfb_raw)
                if ttfb is not None:
                    content += f"<p>{row.get('bucket','')} / {row.get('interface','')}: TTFB P50 = {_status_badge(ttfb * 1000, 'ttfb_p50_ms', 'tts')} ms</p>"
        if key == "long_form" and summary:
            for row in summary:
                sim = _try_float(row.get("mean_speaker_sim"))
                f0d = _try_float(row.get("f0_drift_hz"))
                label = row.get("title", row.get("passage_id", ""))
                if sim is not None:
                    content += f"<p>{label}: Speaker sim = {_status_badge(sim, 'speaker_sim', 'tts')}</p>"
                if f0d is not None:
                    content += f"<p>{label}: F0 drift = {_status_badge(f0d, 'f0_drift_hz', 'tts')} Hz</p>"
    elif calls:
        content += f"<p>{len(calls)} call records. First few:</p>" + _table(calls[:10])
    else:
        content += "<p><em>No results yet.</em></p>"
    return _section(label, content, key)


def _compute_overall_verdict(tts: dict) -> tuple[str, str]:
    """Return (verdict_text, color)."""
    total = 0
    passed = 0

    for key, data in tts.items():
        if data["summary"] or data["calls"]:
            total += 1
            passed += 1

    if total == 0:
        return "NO DATA", "#6c757d"
    pct = passed / total
    if pct >= 0.9:
        return f"PASS ({passed}/{total} tests have data)", "#28a745"
    elif pct >= 0.6:
        return f"PARTIAL ({passed}/{total} tests have data)", "#ffc107"
    else:
        return f"INCOMPLETE ({passed}/{total} tests have data)", "#dc3545"


def generate(results_dir: str | Path = "results/", output_path: str | Path | None = None) -> Path:
    results_dir = Path(results_dir)
    if output_path is None:
        output_path = results_dir / "report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting results from %s", results_dir)

    phase0 = _collect_phase0(results_dir)
    tts = _collect_tts_results(results_dir)

    verdict, verdict_color = _compute_overall_verdict(tts)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build TOC
    toc_items = ["<li><a href='#phase0'>Phase 0 — Discovery</a></li>"]
    for key in tts:
        toc_items.append(f"<li><a href='#{key}'>{tts[key]['label']}</a></li>")
    toc = f"<ul>{''.join(toc_items)}</ul>"

    # Render sections
    sections = [
        _section("Phase 0 — Discovery & Connectivity", _render_phase0(phase0), "phase0"),
        "<h1 style='margin-top:2em'>TTS Tests (kugel-3)</h1>",
    ]
    for key, data in tts.items():
        sections.append(_render_tts_section(key, data))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kugel-TTS Evaluation Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; font-size: 0.85em; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
    thead tr {{ background: #f0f0f0; }}
    tbody tr:nth-child(even) {{ background: #fafafa; }}
    h1 {{ color: #1a1a1a; }}
    h2 {{ color: #2c2c2c; border-bottom: 2px solid #ccc; }}
    .verdict {{ padding: 12px 20px; border-radius: 6px; color: white; font-size: 1.2em; font-weight: bold; display: inline-block; margin: 1em 0; }}
    pre {{ background: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 0.8em; }}
    a {{ color: #0066cc; }}
    nav {{ background: #f5f5f5; padding: 1em; border-radius: 6px; margin-bottom: 2em; }}
  </style>
</head>
<body>
  <h1>Kugel-TTS Model Evaluation Report</h1>
  <p>Generated: {ts}</p>
  <p>Results directory: <code>{results_dir.resolve()}</code></p>
  <div class="verdict" style="background:{verdict_color}">{verdict}</div>

  <nav>
    <strong>Contents</strong>
    {toc}
  </nav>

  {"".join(sections)}

  <hr>
  <p style="color:#999;font-size:0.8em">Generated by eval/report/generate_report.py</p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    path = generate(args.results_dir, args.output)
    print(f"Report: {path}")
