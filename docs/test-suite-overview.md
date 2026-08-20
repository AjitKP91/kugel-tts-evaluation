# TTS Model Evaluation — Test Suite Overview

An automated harness that evaluates a **Text-to-Speech (TTS)** model. Both the
streaming and non-streaming (batch) interfaces are tested, and **all
measurements are client-side** — no access to server internals is assumed.

## Phase 0 — Discovery
Connectivity checks, API-key validation, response-schema discovery, smoke
tests, and cold-start latency measurement against the streaming and batch
endpoints.

## TTS Evaluation — 8 Tests

| # | Test | What it measures |
|---|------|-----------------|
| 2.1 | Naturalness | UTMOS and DNSMOS (OVRL/SIG/BAK) on 200 synthesized sentences |
| 2.2 | Intelligibility | Round-trip WER: synthesize → transcribe with Whisper large-v3 → compare, across 5 sentence categories (Harvard, technical, numbers-basic, numbers-edge, conversational) |
| 2.3 | Prosody | F0 mean/std/range, speaking rate (WPM), rhythm (nPVI) — reference-free |
| 2.4 | Signal Quality | Mel-cepstral distortion (MCD), DTW-aligned, against a matched-speaker reference. When the reference speaker is produced by voice cloning, this measures clone fidelity (how faithfully the engine reproduces the reference speaker). PESQ/STOI are not reported — they are intrusive sample-aligned metrics that don't apply to non-parallel TTS (intelligibility is covered by Test 2.2). Self-skips when no matched-speaker reference is available |
| 2.5 | Streaming Latency | Time-to-first-byte and RTF across 5 text-length buckets × 2 interfaces (streaming vs batch) |
| 2.6 | Throughput & Concurrency | Requests/sec, P50/P99 latency, error rate at N = 1, 5, 10, 20 concurrent |
| 2.7 | Edge Cases | ~100 cases across **17 categories** (see below) |
| 2.8 | Long-Form Consistency | Speaker-similarity drift (ECAPA-TDNN), F0 drift, and speaking-rate drift across multi-paragraph passages |

**Edge-case categories (2.7):** empty/whitespace, single word, very long text,
numbers, punctuation, abbreviations, proper nouns, markup, special characters,
repetition, mixed case, technical/domain terms, questions & commands, lists,
URLs & emails, code snippets, and boundary lengths. Markup cases are treated as
plain-text robustness probes — a pass means the tag characters were handled
without erroring, not that markup directives were interpreted.

## Key Design Principles
- **Both interfaces** — every latency-sensitive test runs on both the streaming and batch interfaces for direct comparison.
- **Client-side only** — all metrics measured in the harness; no server access required.
- **Idempotent resume** — every API call is logged immediately, so an interrupted run resumes where it stopped.
- **Local evaluation models** — a local GPU runs the scoring tools (Whisper for round-trip WER, UTMOS/DNSMOS for naturalness, ECAPA-TDNN for speaker similarity); the model under test runs remotely.
