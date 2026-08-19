# Kugel-TTS Model Evaluation

Automated evaluation harness for KugelAudio's Text-to-Speech model:

| Provider | Type | Model ID |
|----------|------|----------|
| KugelAudio (`kugelaudio.com`) | Text-to-Speech | `kugel-3` |

KugelAudio is a **REST + WebSocket** API (no gRPC). All measurements are made **client-side** — no access to server internals is assumed. Results land under `results/run-<DD-MM-YY>/kugel/` so back-to-back runs don't collide.

> **History note.** This harness began as a multi-provider evaluation for NVIDIA Riva (Parakeet STT + Magpie TTS on SAP AI Core) and Google Gemini-TTS. This repo is the **Kugel-only** cut: STT has been removed (KugelAudio has no ASR product) and the provider abstraction (`eval/tts/client_factory.py`) dispatches to a KugelAudio client. The `provider` switch in `eval/config.yaml` still supports `riva` for the dormant Magpie path.

---

## What It Tests

### Phase 0 — Discovery
Connectivity checks, API-key validation, response-schema discovery, smoke tests, and cold-start measurement against the KugelAudio REST + WebSocket endpoints.

### TTS Evaluation — 8 Tests

| Test | What it measures |
|------|-----------------|
| 2.1 Naturalness | UTMOS and DNSMOS (OVRL/SIG/BAK) on 200 synthesized sentences |
| 2.2 Intelligibility | Round-trip WER: synthesize → Whisper large-v3 → jiwer, across 5 sentence categories |
| 2.3 Prosody | F0 mean/std/range, speaking rate (WPM), rhythm (nPVI) — reference-free |
| 2.4 Signal Quality | MCD / PESQ / STOI against a **matched-speaker reference**. Enabled by cloning a single-speaker corpus (LJSpeech) into a Kugel voice via `scripts/clone_reference_voice.py`; then measures **clone fidelity** (how faithfully Kugel reproduces the cloned speaker). Self-skips until the reference set is built. |
| 2.5 Streaming TTFB & RTF | Time-to-first-byte and real-time factor across 5 text-length buckets × 2 interfaces (**WebSocket** streaming vs REST chunked) |
| 2.6 Throughput & Concurrency | RPS, P50/P99, error rate at N=1,5,10,20 concurrent requests (thread-pool + async REST) |
| 2.7 Edge Cases | ~100 test cases across 17 categories: empty input, numbers, punctuation, Unicode, very long text, etc. (SSML/markup cases treated as plain-text robustness probes — Kugel has no documented SSML) |
| 2.8 Long-Form Consistency | ECAPA-TDNN speaker-similarity drift, F0 drift, speaking-rate drift across multi-paragraph passages |

---

## Project Structure

```
kugel-tts-evaluation/
├── eval/
│   ├── config.yaml           # Provider, endpoints, model/voice — edit before running
│   ├── config.py             # Typed config dataclasses (incl. KugelTTSConfig)
│   ├── utils.py              # Shared: WER normalization, JSONL I/O, stats, retry
│   ├── run.py                # CLI entry point
│   ├── phase0/
│   │   └── discovery.py      # Phase 0 discovery + smoke tests (TTS-only)
│   ├── tts/
│   │   ├── client_factory.py # Provider dispatch (kugel | riva)
│   │   ├── kugel_client.py   # KugelTTSClient: REST batch/stream + WebSocket streaming
│   │   ├── client.py         # TTSClient (Riva/Magpie path — dormant, kept for symmetry)
│   │   └── naturalness.py … long_form.py        # Tests 2.1–2.8
│   ├── data/
│   │   ├── tts_test_sets.py  # All TTS test sentences, edge cases, passages (bundled)
│   │   ├── harvard_sentences.txt
│   │   └── download_datasets.py  # Optional LJSpeech reference downloader
│   └── report/
│       └── generate_report.py    # HTML report with pass/fail badges
├── results/                  # All output written here (created on first run)
├── docs/
│   ├── kugel/
│   │   └── kugel-tts-evaluation-plan.md   # Enablement + run guide
│   └── test-suite-overview.md             # Vendor-neutral test-suite summary
├── requirements.txt
└── RUNBOOK.md                # Step-by-step setup and run guide for the Azure VM
```

---

## Quick Start

**Run this on the Azure VM** (Germany West Central), not a laptop, so network jitter stays out of the latency measurements. See `RUNBOOK.md` for the full setup guide.

```bash
# 1. One-time setup (system packages, venv, dependencies)
bash scripts/setup.sh

# 2. Get a KugelAudio API key from your KugelAudio dashboard, then export it
export KUGELAUDIO_API_KEY="..."

# 3. Set your voice_id / model_id in eval/config.yaml (defaults: voice_id 1071, kugel-3)

# 4. Verify connectivity
python -m eval.run phase0

# 5. Run everything
python -m eval.run all

# 6. Open the report
open results/report.html
```

Run a single test:
```bash
python -m eval.run tts --test naturalness
python -m eval.run tts --test latency
```

Validate config without making API calls:
```bash
python -m eval.run all --dry-run
```

---

## Outputs

Every test writes results immediately to `results/<suite>/<test>/`:
- `calls.jsonl` — one record per API call (used for idempotent resume)
- `summary.csv` — aggregated metrics
- `*.wav` — synthesized audio files (deleted after scoring)

After all tests complete, `results/report.html` contains a full HTML report with per-test tables and pass/fail badges.

---

## Key Design Decisions

**Provider abstraction** — test modules build their client via `eval/tts/client_factory.py`, which dispatches on `tts.provider` in the config. The KugelAudio client mirrors the same public surface as the original Riva client, so the tests are provider-agnostic.

**Idempotent resume** — every API call is written to JSONL immediately. Re-running a test skips already-completed items, so a crashed run picks up where it left off.

**Both interfaces** — every latency-sensitive test runs on both WebSocket (streaming) and REST so the two can be compared directly.

**Client-side only** — all metrics are measured in the harness. No server access required.

**Local GPU for evaluation tools** — the model under test runs remotely on KugelAudio. The VM GPU is used only for local evaluation tools: Whisper large-v3 (round-trip WER), UTMOS/DNSMOS (naturalness scoring), and SpeechBrain ECAPA-TDNN (speaker similarity).

---

## Dependencies

Key packages (see `requirements.txt` for the full list):

| Package | Used for |
|---------|----------|
| `websocket-client` | WebSocket streaming to KugelAudio (TTS 2.5) |
| `requests` / `aiohttp` | REST batch + async concurrency (TTS 2.6) |
| `openai-whisper` | Round-trip WER transcription (TTS 2.2, 2.7) |
| `jiwer` | WER computation |
| `speechbrain` | ECAPA-TDNN speaker embeddings (TTS 2.8) |
| `pyworld` | F0 extraction (TTS 2.3, 2.8) — optional; test runs without it |
| `librosa` | Audio loading, MFCC, duration |
| `datasets>=2.19,<3` | Optional LJSpeech reference (TTS 2.4) |
| `nisqa` / `pesq` / `pystoi` / `dtw-python` | Signal-quality metrics |

---

## Documentation

- `docs/kugel/kugel-tts-evaluation-plan.md` — enablement, configuration, and step-by-step run guide
- `docs/test-suite-overview.md` — vendor-neutral summary of the test suite
- `RUNBOOK.md` — VM setup, configuration, and run instructions
