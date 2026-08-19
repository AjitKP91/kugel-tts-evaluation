# Voice Agent Evaluation: Pipeline, STT, and TTS

A comprehensive reference for measuring pipeline-based voice agents and their constituent STT and TTS models.
Synthesized from industry research, framework docs, and published benchmarks (April 2026).

---

## Table of Contents

1. [Pipeline-Based Voice Agent Evaluation](#1-pipeline-based-voice-agent-evaluation)
   - [1.1 Architecture Overview](#11-architecture-overview)
   - [1.2 Per-Stage KPIs](#12-per-stage-kpis)
   - [1.3 End-to-End Latency KPIs](#13-end-to-end-latency-kpis)
   - [1.4 Turn-Taking & Conversation Quality KPIs](#14-turn-taking--conversation-quality-kpis)
   - [1.5 Interruption & Barge-in KPIs](#15-interruption--barge-in-kpis)
   - [1.6 Quality & Business KPIs](#16-quality--business-kpis)
   - [1.7 Concurrency & Scalability KPIs](#17-concurrency--scalability-kpis)
   - [1.8 Session-Level Cost KPIs](#18-session-level-cost-kpis)
   - [1.9 Industry Benchmarks & Thresholds](#19-industry-benchmarks--thresholds)
   - [1.10 Frameworks, SDKs & Tools](#110-frameworks-sdks--tools)
   - [1.11 Per-Session Measurement Checklist](#111-per-session-measurement-checklist)
   - [1.12 LiveKit Framework Coverage Analysis](#112-livekit-framework-coverage-analysis)
   - [1.13 NVIDIA Riva SDK Coverage Analysis](#113-nvidia-riva-sdk-coverage-analysis)
2. [STT / ASR Evaluation](#2-stt--asr-evaluation)
   - [2.1 Core Accuracy Metrics](#21-core-accuracy-metrics)
   - [2.2 Performance & Latency Metrics](#22-performance--latency-metrics)
   - [2.3 Evaluation Dimensions](#23-evaluation-dimensions)
   - [2.4 Tooling & Libraries](#24-tooling--libraries)
   - [2.5 Benchmark Datasets](#25-benchmark-datasets)
   - [2.6 Model Benchmark Scores](#26-model-benchmark-scores)
   - [2.7 Leaderboards](#27-leaderboards)
   - [2.8 Best Practices](#28-best-practices)
3. [TTS / Speech Synthesis Evaluation](#3-tts--speech-synthesis-evaluation)
   - [3.1 Automated Metrics](#31-automated-metrics)
   - [3.2 Human Evaluation Protocols](#32-human-evaluation-protocols)
   - [3.3 Evaluation Dimensions](#33-evaluation-dimensions)
   - [3.4 Benchmark Datasets](#34-benchmark-datasets)
   - [3.5 Model Benchmark Scores](#35-model-benchmark-scores)
   - [3.6 Leaderboards](#36-leaderboards)
   - [3.7 Consolidated Thresholds Table](#37-consolidated-thresholds-table)
   - [3.8 End-to-End Evaluation Pipeline](#38-end-to-end-evaluation-pipeline)
   - [3.9 Tooling](#39-tooling)
   - [3.10 Evaluation Checklist](#310-evaluation-checklist)
4. [Appendix](#appendix)
   - [A. Pipeline Metric Selection Quick Reference](#a-pipeline-metric-selection-quick-reference)
   - [B. STT Metric Selection Quick Reference](#b-stt-metric-selection-quick-reference)
   - [C. TTS Metric Selection Quick Reference](#c-tts-metric-selection-quick-reference)
   - [D. Key Principles](#d-key-principles)
5. [Sources](#sources)

---

## 1. Pipeline-Based Voice Agent Evaluation

### 1.1 Architecture Overview

```
User Audio ──► VAD ──► STT/ASR ──► EOU Detection ──► LLM ──► TTS ──► Audio Playback
               │         │              │              │        │
             t_vad    t_stt         t_eou          t_llm    t_tts
```

| Stage | Component | Function |
|:---|:---|:---|
| **Ears** | VAD (Voice Activity Detection) | Detect speech onset/offset |
| **Ears** | STT/ASR (Speech-to-Text) | Convert audio to transcript |
| **Brain** | EOU (End-of-Utterance) | Decide when user finished their turn |
| **Brain** | LLM (Large Language Model) | Generate response text |
| **Voice** | TTS (Text-to-Speech) | Synthesize response audio |
| **Transport** | Audio Playback / WebRTC | Deliver audio to user |

Every inter-stage handoff adds latency. The primary goal is minimising perceived response delay while maintaining natural turn-taking.

---

### 1.2 Per-Stage KPIs

#### VAD (Voice Activity Detection)

| Metric | Definition | Formula / Target |
|:---|:---|:---|
| `idle_time` | Time the VAD spent not performing inference | — |
| `inference_duration_total` | Total wall-clock time on VAD inference | — |
| `inference_count` | Number of VAD inference operations | — |
| **VAD Activation Delay** | Time from actual speech onset to VAD trigger | `t_vad_trigger − t_speech_onset` — target **< 30 ms** |
| **VAD Deactivation Delay** | Silence wait before declaring speech ended | Tunable: **200–500 ms** |
| Inference latency per frame | Per-frame decision time (10–32 ms audio chunks) | — |

VAD fires two key events: `speech_start` (onset reference) and `end_of_speech` (reference point for all downstream EOU metrics).

#### STT / ASR Stage

| Metric | Definition | Formula / Target |
|:---|:---|:---|
| `audio_duration` | Duration of audio submitted | — |
| `duration` | Time to produce transcript (0 for streaming) | — |
| `streamed` | Whether STT operates in streaming/WebSocket mode | boolean |
| `acquire_time` | Time to acquire the WebSocket connection | — |
| `connection_reused` | Whether a pooled connection was reused | boolean |
| `transcription_delay` | Time from end-of-speech (VAD) to final transcript | — |
| **STT Streaming Latency** | Partial result delay per audio chunk | `t_partial_result − t_audio_chunk_sent` — **< 100 ms** |
| **STT Finalization Latency** | Final transcript delay after last audio frame | `t_final_transcript − t_last_audio_frame` — **< 300 ms** |
| **TTFS** (Time to Final Segment) | Duration from user silence to final stable transcript | `t_final_transcript − t_user_silence` — **< 400 ms** |
| WER (Word Error Rate) | `(S + D + I) / N × 100%` | — |
| RTF (Real-Time Factor) | `t_processing / t_audio_duration` | Must be **< 1.0** for real-time; target **< 0.3** |

#### EOU (End-of-Utterance) Detection

| Metric | Definition | Formula / Target |
|:---|:---|:---|
| `end_of_utterance_delay` | Time from VAD end-of-speech to committing user's turn | — |
| `transcription_delay` | Subset of EOU delay: waiting for final STT result | — |
| `on_user_turn_completed_delay` | Time to execute the turn-completed callback | — |
| **EOU Precision** | Fraction of triggered EOUs that are real end-of-turn | `TP / (TP + FP)` — target **> 95%** |
| **EOU Recall** | Fraction of real end-of-turns that are detected | `TP / (TP + FN)` — target **> 92%** |
| **False Endpoint Rate** | Fraction of triggered EOUs that are false | `FP / (TP + FP)` — target **< 5%** |
| Turn detector per-turn latency | Model inference time to classify turn completion | 50–160 ms on CPU |

**LiveKit MultilingualModel benchmark (Qwen2.5-0.5B, 396 MB):**

| Metric | Score | Notes |
|:---|:---|:---|
| True Positive Rate | 99.3–99.4% | Correctly fires at real end-of-turn |
| True Negative Rate | 85–96% | English: 87%, Hindi: 96.3% |
| Per-turn inference latency | 50–160 ms | On shared CPU process |

**EOU Detection Methods:**

| Method | Approach | Typical Delay |
|:---|:---|:---|
| Fixed silence threshold | Wait N ms of silence | 500–800 ms |
| Semantic EOU model | Classify text + prosody for turn completion | 200–400 ms |
| Hybrid (VAD + semantic) | Short silence + LLM confidence check | 250–500 ms |

VAD-only endpointing defaults (LiveKit Agents): `min_delay` 500 ms, `max_delay` 3,000 ms, `prefix_padding_ms` 300 ms. OpenAI Realtime API `silence_duration_ms` default: 200 ms.

#### LLM Stage

| Metric | Definition | Target |
|:---|:---|:---|
| `ttft` (Time to First Token) | Time from prompt received to first output token | **< 300 ms** |
| `duration` | Total time from prompt to last token | — |
| **Token Throughput (TPS)** | `n_tokens / t_generation` | **> 50 TPS** |
| `completion_tokens` | Number of output tokens generated | — |
| `prompt_tokens` | Number of input tokens | — |
| `prompt_cached_tokens` | Tokens served from provider's KV cache (reduces TTFT) | — |
| `total_tokens` | `prompt_tokens + completion_tokens` | — |
| `cancelled` | Whether generation was cancelled (barge-in) | boolean |
| `llm_node_ttft` | TTFT as measured at the pipeline node boundary | — |

#### TTS Stage

| Metric | Definition | Target |
|:---|:---|:---|
| `ttfb` (Time to First Byte) | Time from first text token to first audio chunk | **< 150 ms** |
| `duration` | Total synthesis wall-clock time | — |
| `audio_duration` | Duration of synthesized audio output | — |
| `characters_count` | Character count of input text (for billing) | — |
| `streamed` | Whether TTS streams audio incrementally | boolean |
| `acquire_time` | WebSocket connection acquisition time | — |
| `connection_reused` | Whether a pooled WebSocket was reused | boolean |
| `tts_node_ttfb` | TTFB as measured at the pipeline node boundary | — |
| **TTS RTF** | `synthesis_duration / audio_duration` | **< 0.5** (must be < 1.0) |

#### Interruption / Barge-in Stage

| Metric | Definition | Formula / Target |
|:---|:---|:---|
| `detection_delay` | Time from overlapping speech onset to classification | — |
| `prediction_duration` | Model-side inference time only | — |
| `total_duration` | Round-trip time for the interruption inference call | — |
| `num_interruptions` | True barge-ins detected in a session | — |
| `num_backchannels` | Non-interrupting events (e.g., "uh-huh") correctly ignored | — |
| `num_requests` | Total requests to the interruption model | — |
| **Barge-in Detection F1** | Combined precision/recall | `2(Prec×Rec)/(Prec+Rec)` — target **F1 > 0.90** |
| False Positive Rate | VAD-triggered stops that were not genuine interruptions | — |
| False Negative Rate | Genuine barge-ins the agent failed to respond to | — |

---

### 1.3 End-to-End Latency KPIs

**Core formula:**
```
e2e_latency = eou.end_of_utterance_delay + llm.ttft + tts.ttfb
```

**Full pipeline formula:**
```
e2e_full = t_vad_off + t_stt_final + t_eou + t_llm_ttft + t_tts_ttfb + t_transport
```

**Mouth-to-ear latency** (measured via stereo recording + waveform analysis):
```
mouth_to_ear = t_agent_audio_heard − t_user_speech_end_actual
```

**Perceived response latency:**
```
perceived = e2e_latency + (network_RTT / 2)
```

**Percentile targets:**

| Percentile | Target | Interpretation |
|:---|:---|:---|
| P50 | < 600 ms | Median — "snappy" |
| P90 | < 1,000 ms | Most users get acceptable speed |
| P95 | < 1,500 ms | 5% experience noticeable delay |
| P99 | < 2,500 ms | Worst case still usable |

---

### 1.4 Turn-Taking & Conversation Quality KPIs

| Metric | Formula | Target |
|:---|:---|:---|
| **Turn Gap** | `t_agent_audio_start − t_user_speech_end` | **< 800 ms** |
| **Overlap Duration** | `min(t_user_end, t_agent_end) − max(t_user_start, t_agent_start)` | **< 200 ms** |
| **Turns Per Minute (TPM)** | `n_turns / t_session_minutes` | **8–15 TPM** |
| **Double-Talk Ratio** | `Σ t_overlap / t_session_total` | **< 5%** |
| **Turn Completion Rate** | `n_complete_turns / n_total_turns` | **> 95%** |
| Turn detection accuracy | % of utterances correctly identified as complete vs. incomplete | — |
| False interruption rate | Fraction of agent stops triggered by backchannels/noise | — |
| Barge-in latency | Time from user interruption onset to agent stop | — |
| Interruption recovery time | Time for agent to resume after a false interruption | — |

---

### 1.5 Interruption & Barge-in KPIs

| Metric | Formula / Target |
|:---|:---|
| Barge-in Detection F1 | `2(Prec×Rec)/(Prec+Rec)` — **F1 > 0.90** |
| **Barge-in Reaction Time** | `t_agent_audio_stopped − t_user_barge_in_start` — **< 300 ms** |
| **False Activation Rate** | `n_false_stops / n_total_interruptions` — **< 3%** |
| **Barge-in Recovery Time** | `t_new_response_start − t_barge_in_processed` — **< 500 ms** |
| **Backchanneling Rejection** | `n_correctly_ignored / n_backchannels` — **> 90%** |

---

### 1.6 Quality & Business KPIs

| Metric | Target |
|:---|:---|
| **Task Success Rate (TSR)** | > 80% |
| **First Call Resolution (FCR)** | > 85% |
| **Containment Rate** | > 70% |
| **Hallucination Rate** | < 2% |
| **ASR Re-synthesis Accuracy** | < 5% WER |
| **Response Relevance (LLM-as-Judge)** | > 4.0 / 5.0 |
| Task completion rate | % of sessions where agent completed its intended task |
| Intent accuracy | % of user intents correctly understood and acted upon |
| Conversation success rate | % of sessions rated successful by human evaluators |

---

### 1.7 Concurrency & Scalability KPIs

| Metric | Target / Definition |
|:---|:---|
| Concurrent sessions per server | Typical: 10–25 per 4-core/8 GB instance |
| CPU load threshold | Default 70% before refusing new jobs |
| Memory per session | ~2.8 GB / 30 concurrent sessions (4-core/8 GB) |
| **P99 latency under load** | **< 3× P50 idle latency** |
| **Session stability** | **> 99.5%** (no drops) |
| **GPU/CPU utilization ceiling** | **< 80%** at target concurrency |
| Session establishment time | Time from dispatch to agent joining room |
| Cold start latency | Time for a new worker process to be ready |
| WebSocket acquire time | Time to acquire STT/TTS WebSocket connection |
| Connection reuse rate | % of STT/TTS calls using pooled connections |
| Graceful shutdown drain time | Time to complete in-flight sessions (~10+ min for voice) |

**Published load test (LiveKit, 4-core/8 GB):** 30 concurrent sessions — peak ~3.8 cores, ~2.8 GB RAM (~7–8 sessions per core for full STT + VAD + TTS pipeline).

Track P50/P90/P95/P99 for STT, LLM TTFT, TTS TTFB, and E2E at 1×, 2×, 5×, 10× baseline concurrency.

---

### 1.8 Session-Level Cost KPIs

**Cost formula:**
```
cost_session = cost_stt + cost_llm + cost_tts + cost_infra

  cost_stt   = audio_minutes × rate_stt_per_min
  cost_llm   = (input_tokens × rate_in) + (output_tokens × rate_out)
  cost_tts   = characters × rate_tts_per_char
  cost_infra = session_seconds × rate_compute_per_sec
```

**Cost KPIs:**

| Metric | Formula |
|:---|:---|
| Cost per Session | Σ (STT + LLM + TTS + infra) |
| Cost per Turn | `cost_session / n_turns` |
| Cost per Minute | `cost_session / session_minutes` |
| Cost per Resolution | `Σ cost_sessions / n_resolved` |

**Indicative pricing (April 2026):**

| Component | Provider | Price (approx.) |
|:---|:---|:---|
| STT | Deepgram Nova-3 | $0.0043 / min |
| STT | OpenAI Whisper API | $0.006 / min |
| STT | AssemblyAI | $0.0065 / min |
| LLM | GPT-4o-mini | $0.15 / $0.60 per 1M tokens (in/out) |
| LLM | Groq (Llama 3) | $0.05 / $0.08 per 1M tokens |
| TTS | ElevenLabs | $0.18 / 1K chars |
| TTS | Cartesia Sonic | $0.042 / 1K chars |
| TTS | Deepgram Aura | $0.015 / 1K chars |

**Typical session** (3-min call, ~15 turns): **$0.05–$0.25**

**Per-session usage metrics** (track for cost attribution):

| Metric | Scope |
|:---|:---|
| `input_tokens`, `output_tokens`, `total_tokens` | LLM cost tracking |
| `prompt_cached_tokens` | Cache hit rate; reduces LLM cost and TTFT |
| `characters_count` | TTS cost (character-based billing) |
| `audio_duration` | STT/TTS cost (duration-based billing) |
| `session_duration` | Realtime model cost (e.g., OpenAI Realtime API) |

---

### 1.9 Industry Benchmarks & Thresholds

#### End-to-End Latency Classification

| Rating | E2E Latency | User Experience |
|:---|:---|:---|
| Excellent | < 500 ms | Like human conversation |
| Good | 500–800 ms | Slightly noticeable, acceptable |
| Acceptable | 800–1,200 ms | Noticeable pause |
| Poor | 1,200–2,000 ms | User frustration begins |
| Unacceptable | > 2,000 ms | Users drop off |

> Human conversational response time: 200–400 ms. Industry target: **sub-500 ms e2e_latency**.

#### Per-Component Thresholds

| Component | Excellent | Good | Acceptable | Poor |
|:---|:---|:---|:---|:---|
| VAD activation | < 20 ms | < 50 ms | < 100 ms | > 100 ms |
| STT finalization | < 200 ms | < 400 ms | < 600 ms | > 600 ms |
| EOU detection | < 200 ms | < 400 ms | < 600 ms | > 800 ms |
| LLM TTFT | < 200 ms | < 400 ms | < 600 ms | > 800 ms |
| TTS TTFB | < 100 ms | < 200 ms | < 300 ms | > 400 ms |
| Turn gap | < 500 ms | < 800 ms | < 1,200 ms | > 1,500 ms |
| Barge-in reaction | < 200 ms | < 400 ms | < 600 ms | > 800 ms |

#### Noise Cancellation WER Benchmark (Deepgram Nova-3 on noisy gym audio)

| Condition | WER |
|:---|:---|
| Original noisy audio | 117.6% |
| Krisp BVC | 23.5% |
| ai-coustics Voice Focus | 11.8% |

#### Published Load-Test & Provider Results

| Source | Finding |
|:---|:---|
| Deepgram (2025) | Nova-3: ~200 ms median streaming latency at 1,000+ concurrent |
| Cartesia (2025) | Sonic 2: 75 ms TTS TTFB; scales to 10K concurrent |
| LiveKit (2025) | P50 E2E < 800 ms across cloud fleet |
| Hamming AI (2025) | Production agents: P50 turn gap ~650 ms; P99 ~1,800 ms |

#### Provider Latency Claims

| Provider / Model | Claimed Metric |
|:---|:---|
| Deepgram Nova-3 (streaming STT) | < 300 ms transcription latency |
| Cartesia Sonic (TTS) | TTFB 70–120 ms (streaming) |
| ElevenLabs Turbo v2 (TTS) | TTFB ~200 ms |
| OpenAI Realtime API | < 300 ms speech-to-first-audio (ideal conditions) |
| Groq (LLM) | TTFT < 100 ms (small models), 200–400 ms (large) |

---

### 1.10 Frameworks, SDKs & Tools

#### Pipeline Agent Frameworks

| Framework | Language | Key Metrics Surfaced | Notes |
|:---|:---|:---|:---|
| **LiveKit Agents** | Python, Node.js | `EOUMetrics`, `LLMMetrics`, `TTSMetrics`, `STTMetrics`, `VADMetrics`, `InterruptionMetrics`, `e2e_latency`, OTel export | Open source, production-grade |
| **Pipecat** | Python | `enable_metrics=True`, `MetricsLogObserver`, `UserBotLatencyLogObserver`, TTFB, LLM TTFT, TTS first-chunk, `stt-benchmark` tool | Open source; real-time streaming focused |
| **Vocode** | Python | Response latency, STT/TTS plugin timings | Open source |
| **Bolna** | Python | E2E latency per call, cost tracking | Open source, telephony-focused |
| **Retell AI** | Cloud API | Proprietary latency dashboard | SaaS; webhook-based metrics |
| **Vapi** | Cloud API | Dashboard analytics: call duration, cost, latency | SaaS; API/webhook-based |

#### Observability & Telemetry

| Tool | Role |
|:---|:---|
| **OpenTelemetry (OTel)** | Distributed tracing/spans per pipeline stage — industry standard |
| **LangFuse** | LLM observability (tokens, latency, cost); OTel-compatible |
| **Grafana + Tempo / Prometheus** | Dashboards, alerting, P50/P90/P99 histograms |
| **Datadog / SigNoz** | Full-stack APM with OTel export |
| **LiveKit Cloud Agent Insights** | Unified timeline: transcripts + traces + logs + audio playback (30-day retention) |

#### Testing & Evaluation Platforms

| Platform | Focus |
|:---|:---|
| **Hamming AI** | Load testing, AI personas, compliance (PCI/HIPAA), production monitoring |
| **Coval** | CI/CD regression, simulation-first, auto test-case generation |
| **Bluejay** | Real-world simulation testing of deployed voice agents including audio pipeline |
| **Cekura** | E2E testing + production monitoring for voice AI |
| **Braintrust** | Multi-modal eval, LLM-as-Judge scoring |
| **LiveKit Agents test framework** | Text-based LLM behavioral eval, turn simulation, LLM-as-judge |

#### Waveform Analysis

| Tool | Purpose |
|:---|:---|
| **signalwire/latency_checker** | Stereo waveform analysis for mouth-to-ear latency |
| **Audacity** | Manual waveform inspection |

#### Leaderboards

| Leaderboard | URL | Scope |
|:---|:---|:---|
| HF Open ASR Leaderboard | huggingface.co/spaces/hf-audio/open_asr_leaderboard | STT WER comparison |
| TTS Arena (HF) | huggingface.co/spaces/TTS-AGI/TTS-Arena-V2 | TTS Elo rankings |
| Artificial Analysis | artificialanalysis.ai | LLM speed + TTS quality/cost |
| LMSYS Chatbot Arena | lmarena.ai | LLM quality Elo |

#### LiveKit Agents Canonical Measurement Architecture

```
Per-plugin metrics_collected events
  → EOUMetrics, LLMMetrics, TTSMetrics, STTMetrics, VADMetrics
  → Keyed by speech_id for per-turn correlation

ChatMessage.metrics (MetricsReport)
  → e2e_latency, llm_node_ttft, tts_node_ttfb, transcription_delay

session.usage / session_usage_updated event
  → Cumulative token counts, audio durations, per-model breakdown

OpenTelemetry spans
  → set_tracer_provider() → LangFuse, Jaeger, Honeycomb, etc.
```

---

### 1.11 Per-Session Measurement Checklist

```
Per-Session Instrumentation:
- [ ] VAD: Log speech_start, speech_end timestamps
- [ ] STT: Log partial_result, final_transcript + WER
- [ ] EOU: Log eou_triggered timestamp + confidence
- [ ] LLM: Log prompt_sent, first_token, last_token + token counts
- [ ] TTS: Log text_sent, first_audio_chunk, last_audio_chunk
- [ ] Transport: Log audio_played_start on client
- [ ] Barge-in: Log interruption_detected, agent_audio_stopped
- [ ] Session: Log start, end, n_turns, task_outcome
- [ ] Cost: Compute per-component costs (STT + LLM + TTS + infra)
- [ ] Record stereo audio for offline waveform / mouth-to-ear analysis
```

---

### 1.12 LiveKit Framework Coverage Analysis

LiveKit Agents emits rich per-turn and per-session telemetry via `metrics_collected` events and `ChatMessage.metrics`. The table below maps every KPI in this document's Part 1 to the specific LiveKit field(s) that support it, and flags gaps where external tooling is required.

**Coverage legend:**

| Status | Meaning |
|:---|:---|
| ✅ Full | LiveKit publishes the exact field(s) needed — no extra work required |
| 🔶 Derivable | Can be computed from LiveKit fields with lightweight post-processing |
| ⚠️ Partial | LiveKit provides some, but not all, required data points |
| ❌ Not Covered | Must be measured outside LiveKit (external eval, app-layer logic, or infra tools) |

---

#### VAD KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| `idle_time` | Time VAD spent not inferring | `VADMetrics.idle_time` | ✅ Full | Emitted per `metrics_collected` event |
| `inference_duration_total` | Total wall-clock VAD inference time | `VADMetrics.inference_duration_total` | ✅ Full | — |
| `inference_count` | Number of VAD inference ops | `VADMetrics.inference_count` | ✅ Full | — |
| VAD Activation Delay | `t_vad_trigger − t_speech_onset` | None | ❌ Not Covered | LiveKit fires `speech_start` but doesn't expose `t_speech_onset` (acoustic onset) |
| VAD Deactivation Delay | Configurable silence window before `end_of_speech` | None (configurable param) | ⚠️ Partial | Tunable via `VADOptions.min_silence_duration`; actual delay not logged |

---

#### STT / ASR KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| `audio_duration` | Duration of audio submitted to STT | `STTMetrics.audio_duration` | ✅ Full | — |
| `duration` | Wall-clock time to produce transcript | `STTMetrics.duration` | ⚠️ Partial | Always `0` for streaming STT; meaningful only for batch/non-streaming |
| `streamed` | Whether STT ran in streaming/WebSocket mode | `STTMetrics.streamed` | ✅ Full | — |
| `transcription_delay` | Time from end-of-speech to final transcript | `EOUMetrics.transcription_delay`, `ChatMessage.metrics.transcription_delay` | ✅ Full | Available at both stage and turn level |
| STT Streaming Latency | `t_partial_result − t_audio_chunk_sent` | None | ❌ Not Covered | Not exposed; requires network-level tracing or provider SDK hooks |
| STT Finalization Latency | `t_final_transcript − t_last_audio_frame` | None | ❌ Not Covered | `duration=0` for streaming; no frame-level timestamp |
| TTFS (Time to Final Segment) | `t_final_transcript − t_user_silence` | `EOUMetrics.transcription_delay` | 🔶 Derivable | `transcription_delay` captures the same interval from VAD `end_of_speech` |
| RTF (Real-Time Factor) | `t_processing / t_audio_duration` | `STTMetrics.duration`, `STTMetrics.audio_duration` | ⚠️ Partial | Computable for batch mode only; `duration=0` for streaming breaks formula |
| WER (Word Error Rate) | `(S+D+I)/N × 100%` | None | ❌ Not Covered | Requires reference transcripts; use jiwer + ground-truth labels externally |

---

#### EOU (End-of-Utterance) KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| `end_of_utterance_delay` | VAD end-of-speech → turn committed | `EOUMetrics.end_of_utterance_delay` | ✅ Full | — |
| `transcription_delay` | Subset of EOU delay for STT finalization | `EOUMetrics.transcription_delay` | ✅ Full | — |
| `on_user_turn_completed_delay` | Time to execute turn-completed callback | `EOUMetrics.on_user_turn_completed_delay` | ✅ Full | — |
| EOU Precision | `TP / (TP + FP)` | None | ❌ Not Covered | Requires ground-truth turn labels; not provided by SDK |
| EOU Recall | `TP / (TP + FN)` | None | ❌ Not Covered | Same; needs annotated dataset |
| False Endpoint Rate | `FP / (TP + FP)` | None | ❌ Not Covered | Same; annotation required |

---

#### LLM KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| TTFT | Time from prompt send to first token | `LLMMetrics.ttft`, `ChatMessage.metrics.llm_node_ttft` | ✅ Full | Both per-plugin and per-turn views available |
| `duration` | Total LLM generation time | `LLMMetrics.duration` | ✅ Full | — |
| `tokens_per_second` | Generation throughput | `LLMMetrics.tokens_per_second` | ✅ Full | — |
| `completion_tokens` | Output token count | `LLMMetrics.completion_tokens` | ✅ Full | — |
| `prompt_tokens` | Input token count | `LLMMetrics.prompt_tokens` | ✅ Full | — |
| `prompt_cached_tokens` | Tokens served from prompt cache | `LLMMetrics.prompt_cached_tokens` | ✅ Full | Key for cost optimisation |
| `total_tokens` | Sum of prompt + completion | `LLMMetrics.total_tokens` | ✅ Full | — |
| Token Throughput (TPS) | `n_tokens / t_generation` | `LLMMetrics.tokens_per_second` | ✅ Full | Directly published |
| `cancelled` flag | Whether LLM generation was interrupted | Not in official `LLMMetrics` | ❌ Not Covered | Correlate with `InterruptionMetrics.num_interruptions` indirectly |
| Response Relevance | LLM-as-Judge score (> 4.0/5.0) | None | ❌ Not Covered | Application-level; requires external evaluator (e.g. Braintrust, DeepEval) |

---

#### TTS KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| TTFB | Time from text-in to first audio chunk | `TTSMetrics.ttfb`, `ChatMessage.metrics.tts_node_ttfb` | ✅ Full | Both per-plugin and per-turn views |
| `duration` | Total TTS synthesis time | `TTSMetrics.duration` | ✅ Full | — |
| `audio_duration` | Duration of synthesized audio | `TTSMetrics.audio_duration` | ✅ Full | — |
| `characters_count` | Characters sent to TTS | `TTSMetrics.characters_count` | ✅ Full | Used for cost calculation |
| `streamed` | Whether TTS output was streamed | `TTSMetrics.streamed` | ✅ Full | — |
| `acquire_time` | Time to acquire TTS connection | Not in official `TTSMetrics` | ❌ Not Covered | Not in current SDK spec |
| `connection_reused` | Whether a pooled connection was reused | Not in official `TTSMetrics` | ❌ Not Covered | Not in current SDK spec |
| TTS RTF | `t_synthesis / t_audio_duration` | `TTSMetrics.duration`, `TTSMetrics.audio_duration` | 🔶 Derivable | Compute `duration / audio_duration`; both fields available |
| MOS / UTMOS / PESQ | Perceptual audio quality scores | None | ❌ Not Covered | Requires audio capture + offline evaluation with UTMOS/speechmetrics |

---

#### Interruption / Barge-in KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| `detection_delay` | Time from barge-in to detection | `InterruptionMetrics.detection_delay` | ✅ Full | — |
| `prediction_duration` | Time for interruption classifier to decide | `InterruptionMetrics.prediction_duration` | ✅ Full | — |
| `total_duration` | Cumulative interruption duration | `InterruptionMetrics.total_duration` | ✅ Full | — |
| `num_interruptions` | Count of interruptions in session | `InterruptionMetrics.num_interruptions` | ✅ Full | — |
| `num_backchannels` | Count of backchannels (filler words) | `InterruptionMetrics.num_backchannels` | ✅ Full | — |
| `num_requests` | Total LLM requests (incl. cancelled) | `InterruptionMetrics.num_requests` | ✅ Full | — |
| Barge-in Reaction Time | Time from barge-in to agent stopping | `InterruptionMetrics.detection_delay` | ✅ Full | — |
| Barge-in F1 | `2(Prec×Rec)/(Prec+Rec)` | None | ❌ Not Covered | Requires ground-truth labels for each interruption event |
| False Activation Rate | `FP_interruptions / total_interruptions` | None | ❌ Not Covered | No ground-truth annotation in SDK |
| Backchanneling Rejection Rate | `correctly_ignored / total_backchannels` | None | ❌ Not Covered | `num_backchannels` is a count only; no per-event outcome label |

---

#### End-to-End Latency KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| E2E latency (core) | `eou.end_of_utterance_delay + llm.ttft + tts.ttfb` | `ChatMessage.metrics.e2e_latency` | ✅ Full | Pre-computed by SDK; also derivable component-by-component |
| E2E latency (full pipeline) | `t_vad_off + t_stt_final + t_eou + t_llm_ttft + t_tts_ttfb + t_transport` | `ChatMessage.metrics.e2e_latency` + transport RTT | ⚠️ Partial | Core covered; transport RTT requires client-side measurement |
| Mouth-to-ear latency | `t_agent_audio_heard − t_user_speech_end_actual` | None | ❌ Not Covered | Requires stereo waveform capture (e.g. signalwire/latency_checker); no client-side timestamp |
| Perceived response latency | `e2e_latency + RTT/2` | `ChatMessage.metrics.e2e_latency` + RTT | 🔶 Derivable | `e2e_latency` available; RTT from WebRTC stats (LiveKit room.engine.stats) |
| P50 / P90 / P95 / P99 percentiles | Percentile distribution of `e2e_latency` | `ChatMessage.metrics.e2e_latency` (per turn) | 🔶 Derivable | Collect all per-turn values; compute percentiles in Grafana/Prometheus |
| `end_of_turn_delay` | Time from EOU to agent starting to speak | `ChatMessage.metrics.end_of_turn_delay` | ✅ Full | — |
| `on_user_turn_completed_delay` | Turn-completed callback overhead | `ChatMessage.metrics.on_user_turn_completed_delay` | ✅ Full | — |

---

#### Turn-Taking KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| Turn Gap | `t_agent_audio_start − t_user_speech_end` | `ChatMessage.metrics.started_speaking_at`, `stopped_speaking_at` | 🔶 Derivable | Both timestamps on `ChatMessage.metrics`; compute the delta |
| Overlap Duration | `Σ max(0, t_agent_start − t_user_end)` | `started_speaking_at`, `stopped_speaking_at` | 🔶 Derivable | Cross-reference consecutive ChatMessage.metrics entries |
| Turns Per Minute (TPM) | `n_turns / session_minutes` | Conversation history + session duration | 🔶 Derivable | Count `ConversationItem` events; divide by session wall-clock time |
| Double-Talk Ratio | `Σ t_overlap / t_session_total` | None directly | ❌ Not Covered | No direct overlap measurement; requires audio track analysis |
| Turn Completion Rate | `n_complete_turns / n_total_turns` | None | ❌ Not Covered | Requires application-level turn labelling |
| False Interruption Rate | Agent stops speaking on a non-interruption | `InterruptionMetrics.num_interruptions` | ⚠️ Partial | Count available; distinguishing false from true interruptions needs labels |

---

#### Business KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| Task Success Rate (TSR) | `n_successful_tasks / n_total_tasks` | None | ❌ Not Covered | Application-level; requires task outcome logging (webhook, database) |
| First Call Resolution (FCR) | `n_resolved_first_contact / n_total_contacts` | None | ❌ Not Covered | CRM/telephony layer, not SDK |
| Containment Rate | `n_handled_by_agent / n_total_sessions` | None | ❌ Not Covered | Application/routing layer |
| Hallucination Rate | `n_hallucinated_responses / n_total` | None | ❌ Not Covered | Requires LLM-as-Judge or human review |
| ASR Re-synthesis Accuracy | WER via TTS→ASR roundtrip | None | ❌ Not Covered | External pipeline: synthesize → transcribe → measure WER |
| Response Relevance | LLM-as-Judge score | None | ❌ Not Covered | External evaluator (Braintrust, DeepEval, promptfoo) |

---

#### Concurrency & Scalability KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| Concurrent sessions | Active session count | None (infra-level) | ❌ Not Covered | Monitor via LiveKit Cloud dashboard or Kubernetes HPA metrics |
| Session Stability Rate | `n_clean_sessions / n_total_sessions` | None | ❌ Not Covered | Application-level; detect via error events or session-end reason codes |
| P99 latency ratio | `P99 / P50 ≤ 3×` | `e2e_latency` per turn | 🔶 Derivable | Collect per-turn values; percentile analysis in Prometheus/Grafana |
| GPU / CPU utilisation | Resource % at target concurrency | None | ❌ Not Covered | Infrastructure metrics (Prometheus node_exporter, cAdvisor, Datadog) |

---

#### Session-Level Cost KPIs

| KPI | Formula / Definition | LiveKit Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| STT input duration | Total audio transcribed | `STTModelUsage.audio_duration` (session_usage_updated) | ✅ Full | — |
| LLM input tokens | Prompt tokens incl. cached | `LLMModelUsage.input_tokens`, `LLMModelUsage.input_cached_tokens` | ✅ Full | — |
| LLM output tokens | Completion tokens | `LLMModelUsage.output_tokens` | ✅ Full | — |
| TTS characters | Characters sent to synthesis | `TTSModelUsage.characters_count` | ✅ Full | — |
| TTS audio duration | Duration of generated audio | `TTSModelUsage.audio_duration` | ✅ Full | — |
| Total interruption requests | LLM calls triggered (incl. cancelled) | `InterruptionModelUsage.total_requests` | ✅ Full | — |
| Cost per session | `cost_stt + cost_llm + cost_tts + cost_infra` | Fields above (apply provider rates) | 🔶 Derivable | All inputs available; multiply by provider rate cards |
| Infra cost | Compute/bandwidth cost | None | ❌ Not Covered | Cloud provider billing (AWS, GCP, LiveKit Cloud usage API) |

---

#### Summary

| Coverage Category | KPI Count | Percentage |
|:---|:---:|:---:|
| ✅ Full — directly published | ~35 | ~48% |
| 🔶 Derivable — compute from LiveKit fields | ~12 | ~16% |
| ⚠️ Partial — some data available | ~5 | ~7% |
| ❌ Not Covered — external tooling required | ~22 | ~30% |

**What LiveKit covers well:** All core latency pipeline KPIs (`e2e_latency`, `llm_node_ttft`, `tts_node_ttfb`, per-stage delays), LLM token economics, TTS character/audio usage, interruption counts and timing, and per-turn `started_speaking_at`/`stopped_speaking_at` timestamps.

**Key gaps requiring external tooling:**
- **Audio quality** (WER, MOS, UTMOS, PESQ): requires ground-truth transcripts + offline eval pipelines
- **Precision/Recall KPIs** (EOU accuracy, Barge-in F1): requires annotated ground-truth turn labels
- **Business KPIs** (TSR, FCR, Containment): application/CRM layer
- **Infrastructure** (GPU utilisation, session stability): infra monitoring stack (Prometheus, Datadog)
- **Mouth-to-ear latency**: stereo waveform capture tool (signalwire/latency_checker)

---

### 1.13 NVIDIA Riva SDK Coverage Analysis

NVIDIA Riva is a GPU-accelerated speech AI server (built on Triton Inference Server) with Python and C++ client SDKs. Unlike LiveKit, Riva is a **model-serving layer**, not a pipeline agent framework — it exposes STT and TTS endpoints directly and does not manage EOU, turn-taking, or LLM orchestration. Its coverage of pipeline-level KPIs is therefore scoped to the ASR and TTS stages only, but within those stages it provides strong latency profiling and server-level GPU telemetry.

**Coverage legend:** same as Section 1.12 (✅ Full / 🔶 Derivable / ⚠️ Partial / ❌ Not Covered)

---

#### ASR / STT KPIs — Python SDK Response Fields

| KPI | Formula / Definition | Riva Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| Confidence score | Per-alternative score (0–1) | `SpeechRecognitionAlternative.confidence` | ✅ Full | Populated on final results only (`is_final=true`) |
| Word-level timing | `WordInfo.start_time` / `end_time` (ms) | `WordInfo.start_time`, `end_time` | ✅ Full | Requires `enable_word_time_offsets=true` in config |
| Interim stability | Fraction of interim words not revised | `StreamingRecognitionResult.stability` (0–1) | ✅ Full | — |
| `audio_processed` | Cumulative seconds of audio processed | `StreamingRecognitionResult.audio_processed` | ✅ Full | Use as RTF numerator |
| RTF (Real-Time Factor) | `t_processing / t_audio_duration` | Derived from `audio_processed` + wall-clock | 🔶 Derivable | C++ client computes RTFX directly |
| RTFX | Inverse RTF (`audio_processed / wall-clock`) | `riva_streaming_asr_client` output | ✅ Full | Built-in in C++ perf client |
| Streaming latency P50/P90/P95/P99 | Percentile distribution of final-result latency | `riva_streaming_asr_client` output | ✅ Full | Official C++ load testing tool |
| Partial result latency | `t_partial_result − t_audio_chunk_sent` | Wall-clock delta to `is_final=false` result | 🔶 Derivable | No automatic computation; instrument client-side |
| Finalization latency | `t_final_transcript − t_last_audio_frame` | Wall-clock delta to `is_final=true` result | 🔶 Derivable | Instrument client-side with timestamps |
| Speaker diarization segments | `speaker_tag` per word | `WordInfo.speaker_tag` | ✅ Full | Requires diarization enabled in config |
| VAD probability stream | Frame-level speech probability | `PipelineStates.vad_probabilities[]` | ✅ Full | Useful for VAD quality analysis |
| Confidence calibration (ECE) | Expected Calibration Error | `confidence` per alternative | 🔶 Derivable | Collect confidences + WER labels; compute ECE externally |
| WER | `(S+D+I)/N × 100%` | None | ❌ Not Covered | Transcribe reference set → compute with `jiwer` externally |
| CER | Character-level edit distance | None | ❌ Not Covered | External — same pipeline as WER |
| Noise robustness WER | WER at various SNR levels | None | ❌ Not Covered | External — inject noise with `audiomentations`, eval with `jiwer` |
| DER (Diarization Error Rate) | `(FA + MISS + ERROR) / total` | `WordInfo.speaker_tag` (partial input) | ⚠️ Partial | Word-level tags available; compute DER with `pyannote.metrics` |

---

#### TTS KPIs — Python SDK Response Fields

| KPI | Formula / Definition | Riva Field(s) | Coverage | Notes |
|:---|:---|:---|:---:|:---|
| TTFB (Python) | Wall-clock to first `response.audio` chunk | Client-side timestamp delta | 🔶 Derivable | Implemented in `talk.py`; not a dedicated field |
| TTFB (load test) | P50/P90/P95/P99 first-chunk latency | `riva_tts_perf_client` output | ✅ Full | Built-in in C++ perf client |
| Inter-chunk latency | Delta between successive audio chunks | `riva_tts_perf_client` output | ✅ Full | — |
| TTS RTF | `t_synthesis / t_audio_duration` | `riva_tts_perf_client` output | ✅ Full | Also derivable: `len(audio) / (sample_rate × 2)` vs wall-clock |
| Synthesized audio duration | From raw PCM bytes | `len(resp.audio) / (sample_rate × 2)` | 🔶 Derivable | No explicit duration field in response |
| Phoneme duration (per token) | Predicted duration per token | `SynthesizeSpeechResponseMetadata.predicted_durations[]` | ⚠️ Partial | Experimental field; model support varies |
| MOS / UTMOS | Perceptual audio quality | None | ❌ Not Covered | Capture audio → eval with `UTMOS`, `DNSMOS`, `speechmetrics` |
| PESQ / STOI | Signal quality vs reference | None | ❌ Not Covered | External — `pesq`, `pystoi` on captured audio |
| MCD | Mel-cepstral distortion | None | ❌ Not Covered | External — `pyworld` + DTW alignment |
| Speaker similarity | Cosine similarity of embeddings | None | ❌ Not Covered | External — SpeechBrain ECAPA-TDNN or `resemblyzer` |
| WER round-trip (TTS quality) | ASR(TTS(text)) vs original text | None | ❌ Not Covered | External pipeline: synthesize → transcribe → `jiwer` |

---

#### Server-Level KPIs — Triton Prometheus Metrics (`:8002/metrics`)

| KPI | Prometheus Metric | Coverage | Notes |
|:---|:---|:---:|:---|
| GPU utilisation | `nv_gpu_utilization` | ✅ Full | Per-GPU, 0–1 |
| GPU memory used | `nv_gpu_memory_used_bytes` | ✅ Full | Per-GPU |
| Concurrent requests (queue depth) | `nv_inference_pending_request_count` | ✅ Full | Proxy for concurrent stream saturation |
| Inference E2E latency | `nv_inference_request_duration_us` | ✅ Full | Histogram; per-model labels |
| Model execution time | `nv_inference_compute_infer_duration_us` | ✅ Full | Excludes queue/pre/post |
| Queue wait time | `nv_inference_queue_duration_us` | ✅ Full | Latency under load |
| Throughput (requests/sec) | `nv_inference_request_success` (rate) | ✅ Full | Compute via Prometheus `rate()` |
| Response cache hit rate | `nv_cache_num_hits_per_model` / `nv_cache_num_misses_per_model` | ✅ Full | Per-model labels |
| CPU utilisation | `nv_cpu_utilization` | ✅ Full | — |

---

#### C++ Performance Client Tools

```bash
# ASR load test — streams audio files, prints P50/P90/P95/P99 + RTFX
riva_streaming_asr_client \
  --audio_file=test.wav \
  --num_parallel_requests=50 \
  --num_iterations=100 \
  --simulate_realtime \
  --interim_results \
  --word_time_offsets \
  --print_transcripts

# TTS load test — measures TTFB, inter-chunk latency, RTF
riva_tts_perf_client \
  --text_file=sentences.txt \
  --num_parallel_requests=20 \
  --num_iterations=100 \
  --online   # streaming mode
```

Both are pre-compiled in the official Riva Docker container and available as source at `github.com/nvidia-riva/cpp-clients`.

---

#### Summary

| Coverage Category | STT KPIs | TTS KPIs | Server KPIs |
|:---|:---:|:---:|:---:|
| ✅ Full — directly available | ~7 | ~4 | ~9 |
| 🔶 Derivable — compute from Riva fields | ~4 | ~3 | — |
| ⚠️ Partial — some data only | ~1 | ~1 | — |
| ❌ Not Covered — external tooling needed | ~3 | ~6 | — |

**What Riva covers well:** Latency percentiles (P50–P99) and RTFX for both ASR and TTS via C++ perf clients; word-level timing and confidence scores; GPU utilisation and queue depth via Triton Prometheus metrics; TTFB for TTS streaming.

**Key gaps:** All perceptual audio quality metrics (WER, MOS, UTMOS, PESQ, STOI, MCD, Speaker Similarity) are external. Riva is a model server, not an eval framework — treat it as the inference backend and layer external quality eval on top.

**Recommended stack:**
- Riva C++ clients → latency/throughput benchmarks
- `jiwer` / NeMo eval → WER from Riva transcripts
- `UTMOS` / `speechmetrics` / `resemblyzer` → TTS quality from Riva audio output
- Grafana + Prometheus → Triton server KPI dashboards

---

## 2. STT / ASR Evaluation

### 2.1 Core Accuracy Metrics

All accuracy metrics are grounded in **minimum edit distance (Levenshtein)** between a reference transcript and a model hypothesis.

**Primitives:**
- **S** = Substitutions (wrong word)
- **D** = Deletions (reference word missing)
- **I** = Insertions (extra word in hypothesis)
- **H** = Hits (correct matches)
- **N** = H + S + D (total reference words)

#### WER (Word Error Rate) — Universal Standard

```
WER = (S + D + I) / N  =  (S + D + I) / (H + S + D)
```

| Rating | WER Range | Interpretation |
|:---|:---|:---|
| Excellent | < 3% | Near-human accuracy on clean speech |
| Good | 3–5% | Production-ready for most use cases |
| Acceptable | 5–10% | Usable with post-processing |
| Poor | 10–20% | Significant errors; domain mismatch likely |
| Unusable | > 20% | Not suitable for production |

**Caveats:**
- WER can exceed 100% when insertions are high (e.g., noisy audio)
- Treats all word errors equally ("their" vs "there" = same penalty as content errors)
- Sensitive to normalization (punctuation, casing, number formatting)

#### CER (Character Error Rate)

```
CER = (S_char + D_char + I_char) / N_char
```

Preferred for logographic languages (Chinese, Japanese, Korean) where word boundaries are ambiguous. Typically 2–5× lower than WER for alphabetic languages.

#### MER (Match Error Rate)

```
MER = (S + D + I) / (S + D + I + H)
```

Bounded [0, 1] — cannot exceed 100%. MER ≥ WER always.

#### WIL (Word Information Lost) and WIP (Word Information Preserved)

```
WIP = (H / N) × (H / (H + I))   [precision × recall proxy]
WIL = 1 − WIP
```

Captures both hallucinated words (insertions) and missed words (deletions) simultaneously.

#### SER (Sentence Error Rate)

```
SER = (utterances with ≥1 error) / (total utterances)
```

Critical for voice command / intent-classification systems where any error is a full failure.

#### NE-WER (Named-Entity WER)

```
NE-WER = (S_NE + D_NE + I_NE) / N_NE
```

Compute WER only on domain-critical terms extracted via a NER tagger. Essential for medical, legal, and financial transcription.

#### Punctuation & Formatting Metrics

| Metric | Formula |
|:---|:---|
| **Punctuation F1** | `2 × (Prec × Rec) / (Prec + Rec)` for each punctuation mark |
| **Capitalization Accuracy** | `n_correct_caps / n_total_cap_decisions` |
| **Number Formatting Accuracy** | Accuracy of spoken numbers → written form ("twenty-three" → "23") |

#### Semantic Error Metrics (Emerging)

| Metric | Description |
|:---|:---|
| **SeMaScore** | Semantic similarity between reference and hypothesis using sentence embeddings |
| **LLM-as-Judge** | Use GPT-4/Claude to score transcription quality on semantic meaning |
| **KRR (Keyword Recall Rate)** | % of domain-critical keywords correctly transcribed |
| **Intent Preservation** | Whether ASR errors change the downstream intent classification |

#### BLEU Score (Speech Translation)

Used when ASR also translates (e.g., Whisper's speech-to-text translation). Scored against reference translations. NVIDIA Canary-1B example: En→De 32.15, En→Fr 40.76.

---

### 2.2 Performance & Latency Metrics

#### RTF (Real-Time Factor) and RTFx

```
RTF  = t_processing / t_audio_duration       (lower is better; must be < 1.0 for real-time)
RTFx = t_audio_duration / t_processing       (higher is better; reported on HF Open ASR Leaderboard)
```

| Rating | RTF | RTFx | Suitability |
|:---|:---|:---|:---|
| Excellent | < 0.1 | > 10 | Real-time streaming, voice agents |
| Good | 0.1–0.3 | 3–10 | Streaming with comfortable buffer |
| Acceptable | 0.3–1.0 | 1–3 | Near-real-time, some buffering |
| Too Slow | > 1.0 | < 1 | Batch-only |

| Model Type | Typical RTFx | Notes |
|:---|:---|:---|
| Encoder-decoder (Whisper large-v3) | ~146 | Slower; higher accuracy |
| CTC/TDT (Parakeet-TDT-0.6B-v2) | ~3,386 | ~23× faster; slightly lower accuracy |

#### TTFW (Time to First Word)

Wall-clock time from audio start until first word token is emitted:

| Engine | TTFW |
|:---|:---|
| Azure Speech Real-Time | 530 ms |
| Picovoice Cheetah | 590 ms |
| Moonshine Medium | 640 ms |
| Google STT Streaming | 830 ms |
| Amazon Transcribe Streaming | 920 ms |

#### Streaming Latency Metrics

| Metric | Definition | Target |
|:---|:---|:---|
| **Partial Result Latency** | Delay for intermediate (non-final) transcripts | < 100 ms per chunk |
| **Finalization Latency** | Delay from user silence to final stable transcript | < 300 ms |
| **Endpointing Delay** | Time for STT to declare end-of-utterance | 200–800 ms (model-dependent) |
| **TTFS** (Time to Final Segment) | From user silence → final stable transcript | < 400 ms |
| **Stability Rate** | Fraction of interim words not revised in later stream updates | Higher = better |
| **Trailing Latency** | Average delay between a word being spoken and model emitting it | — |

#### Throughput Metrics

| Metric | Definition |
|:---|:---|
| **Concurrent Stream Capacity** | Max simultaneous audio streams at target RTF |
| **Batch Throughput** | Hours of audio processed per wall-clock hour |
| **GPU Utilization** | Compute efficiency at target concurrency |

---

### 2.3 Evaluation Dimensions

#### Noise Robustness

| Test Condition | Method | Metrics |
|:---|:---|:---|
| Clean speech | Baseline evaluation | WER on clean test sets |
| Additive noise (white, babble, traffic) | Synthetic noise at 5–20 dB SNR | WER degradation curve |
| Reverberation | Simulate room acoustics (RIR convolution) | WER vs. RT60 |
| Far-field capture | 1–3 m microphone distance | WER delta from close-talk |
| Telephony codec artifacts | G.711, Opus, AMR encoding | WER on re-encoded audio |

**Parakeet-TDT-0.6B-v2 on additive white noise:**

| SNR | WER | Relative Degradation |
|:---|:---|:---|
| Baseline (clean) | 6.05% | — |
| 10 dB | 6.95% | +15% |
| 5 dB | 8.23% | +36% |
| 0 dB | 11.88% | +96% |
| −5 dB | 20.26% | +235% |

Key datasets: CHiME-4/5/6/7/8, DEMAND noise corpus, MUSAN, DNS Challenge, Aurora-4.

#### Accent & Dialect Robustness

Compute per-accent WER and report the **fairness delta** (WER gap between accent groups).

| Dimension | Evaluation Approach |
|:---|:---|
| Regional accents | WER per accent group (Southern US, Indian English, Scottish, etc.) |
| Non-native speakers | WER on L2 English speakers vs. native |
| Code-switching | WER on bilingual speech segments |
| Dialectal variation | WER per dialect using Common Voice metadata filters |

**Canary-Qwen-2.5B on CasualConversations-v1 (fairness audit):**

| Group | WER |
|:---|:---|
| Male | 16.71% |
| Female | 13.85% |
| Gender "other" | 29.46% |

Key datasets: CORAAL (AAL dialects), L2-ARCTIC (6 non-native L1s), AESRC2020 (8 accented English), Common Voice.

#### Domain-Specific Vocabulary

| Dimension | Approach |
|:---|:---|
| OOV Rate | Fraction of reference words not in the model's vocabulary |
| KRR (Keyword Recall Rate) | % of domain terms correctly transcribed |
| Custom vocabulary / hotwords | Test with and without injected terms; measure WER delta |
| Domain-specific test sets | Build from production audio with verified transcripts |

Domain examples: medical drug names, legal case citations, financial ticker symbols, technical acronyms. SPGISpeech (financial): ~1.9–3.2% WER for top models. Earnings-22 (harder): 9–15%.

#### Punctuation & Capitalization

| Metric | Formula |
|:---|:---|
| Punctuation F1 | `2 × (Prec × Rec) / (Prec + Rec)` for each mark |
| Capitalization Accuracy | `n_correct_caps / n_total_cap_decisions` |
| Number Formatting | Accuracy of spoken → written number conversion |

#### Speaker Diarization

```
DER = (Miss + False_Alarm + Speaker_Error) / Total_Reference_Duration
```

| Metric | Formula | Target |
|:---|:---|:---|
| **DER** (Diarization Error Rate) | `(Missed + False_Alarm + Confusion) / Reference_Duration` | < 10% |
| **JER** (Jaccard Error Rate) | Per-speaker Jaccard error, averaged across speakers | < 15% |
| **Speaker Count Accuracy** | `correct_count / total_sessions` | > 90% |
| **SA-WER** (Speaker-attributed WER) | WER per speaker after optimal assignment | — |

Standard 250 ms collar applied around speaker boundaries. Tools: `pyannote.metrics`, `dscore`.

#### Confidence Score Calibration

| Evaluation | Method |
|:---|:---|
| **ECE** (Expected Calibration Error) | Mean absolute gap between confidence bucket value and actual accuracy |
| **AUROC for error detection** | Confidence as binary classifier for "will this word be wrong?" |
| **Reliability diagram** | Plot mean confidence vs. mean accuracy per decile |

#### Streaming vs. Batch Comparison

| Dimension | Streaming | Batch |
|:---|:---|:---|
| Latency | Real-time (< 300 ms finalization) | Minutes to hours |
| Accuracy | Typically 1–3% higher WER | Lower WER (full context available) |
| Context window | Limited (sliding window) | Full audio file |
| Use case | Voice agents, live captioning | Post-call analytics, transcription |
| Cost | Per-minute (higher rate) | Often lower volume rates |

Commercial streaming vs. batch WER gap: Amazon +1.3%, Azure +2.7%, Google +3.0%.

#### Long-Form Audio

Models trained on < 30 s segments can hallucinate or repeat when given full-length files. Measure:
- WER on full files vs. pre-segmented chunks
- **Hallucination rate**: tokens generated during silence
- **Repetition rate**: duplicate phrases in output

Distil-Whisper large-v3's sequential algorithm closed a 4.8% WER gap vs. distil-large-v2 on long-form decoding while running 6.3× faster than full Whisper large-v3.

---

### 2.4 Tooling & Libraries

#### Evaluation Libraries

| Tool | Language | Metrics | Usage |
|:---|:---|:---|:---|
| **jiwer** | Python | WER, CER, MER, WIL, WIP | `pip install jiwer` — industry standard |
| **HF evaluate** | Python | WER, CER + 100+ metrics | `evaluate.load("wer")` |
| **NIST sclite / SCTK** | C/Perl | WER, SER, alignments | Official NIST scoring; required for formal evaluations |
| **Kaldi** | C++/Bash | WER, lattice-based metrics | Academic-grade ASR evaluation |
| **pyannote.metrics** | Python | DER, JER, coverage, purity | Diarization evaluation standard |
| **torchmetrics WER** | Python | WER | GPU-accelerated batch evaluation |
| **HF open_asr_leaderboard** | Python | WER + RTFx | Full eval pipeline for 15+ ASR frameworks |

#### ASR Inference Frameworks

| Framework | Purpose |
|:---|:---|
| **OpenAI Whisper** | Open-source ASR model; local inference via `openai-whisper` |
| **faster-whisper** | CTranslate2-optimized; 4× faster, lower memory |
| **whisper.cpp** | C++ port for CPU/edge inference |
| **NVIDIA NeMo** | Enterprise ASR training + evaluation toolkit |
| **SpeechBrain** | Research ASR framework with evaluation recipes |
| **Whisper JAX** | JAX-optimized for TPU/batch inference |

#### Code Examples

**jiwer — full usage:**
```python
import jiwer

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the fast brown cat jumps over the lazy fog"

# Basic metrics
print(f"WER: {jiwer.wer(reference, hypothesis):.2%}")   # 33.33%
print(f"CER: {jiwer.cer(reference, hypothesis):.2%}")
print(f"MER: {jiwer.mer(reference, hypothesis):.2%}")
print(f"WIL: {jiwer.wil(reference, hypothesis):.2%}")

# Detailed breakdown
measures = jiwer.compute_measures(reference, hypothesis)
print(f"Substitutions: {measures['substitutions']}")
print(f"Deletions:     {measures['deletions']}")
print(f"Insertions:    {measures['insertions']}")
print(f"Hits:          {measures['hits']}")

# Alignment visualization
output = jiwer.process_words(reference, hypothesis)
print(jiwer.visualize_alignment(output))

# Batch evaluation (lists of sentences)
refs = ["the quick brown fox", "i love python"]
hyps = ["the fast brown cat", "i like python"]
print(f"Batch WER: {jiwer.wer(refs, hyps):.2%}")
```

**Text normalization pipeline (before computing WER):**
```python
import jiwer

transform = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])

wer = jiwer.wer(reference, hypothesis,
                truth_transform=transform,
                hypothesis_transform=transform)
```

**HF evaluate:**
```python
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

references = ["hello world", "this is a test"]
predictions = ["hello word", "this is test"]

print(f"WER: {wer_metric.compute(predictions=predictions, references=references):.2%}")
print(f"CER: {cer_metric.compute(predictions=predictions, references=references):.2%}")
```

**Noise injection:**
```python
from audiomentations import AddGaussianNoise
import soundfile as sf

augment = AddGaussianNoise(min_snr_in_db=0, max_snr_in_db=0, p=1.0)
audio, sr = sf.read("clean.wav")
noisy = augment(audio, sample_rate=sr)
```

---

### 2.5 Benchmark Datasets

#### Standard English Datasets (ESB — HuggingFace `esb/datasets`)

| Dataset | Domain | Style | Test Hours | Transcriptions |
|:---|:---|:---|:---|:---|
| LibriSpeech test-clean | Audiobooks | Narrated, clear | 5.4 h | Normalized |
| LibriSpeech test-other | Audiobooks | Diverse/challenging | 5.4 h | Normalized |
| TED-LIUM v3 | TED Talks | Prepared oratory | 3 h | Normalized |
| GigaSpeech (test-XL) | Mixed sources | Mixed | 40 h | Punctuated |
| SPGISpeech (test) | Financial earnings | Oratory + spontaneous | 100 h | Punctuated + cased |
| VoxPopuli (en test) | EU Parliament | Oratory, non-native | 5 h | Punctuated |
| AMI (IHM test) | Indoor meetings | Spontaneous | 9 h | Punctuated + cased |
| Earnings-22 (test) | Earnings calls | Spontaneous, global | 5 h | Punctuated + cased |

Loading: `from datasets import load_dataset; ds = load_dataset("esb/datasets", "librispeech")`

#### Multilingual Datasets

| Dataset | Languages | Hours / Language | Notes |
|:---|:---|:---|:---|
| FLEURS | 102 | ~10 h | Read speech; CC-BY-4.0 |
| MLS (Multilingual LibriSpeech) | 8 | 100–44,600 h | LibriVox audiobooks |
| Common Voice | 100+ | 5–2,000 h | Crowd-sourced |
| VoxPopuli | 23 | 400–500 h | EU Parliament; CC0 |
| CoVoST-2 | 21→en | varies | Speech translation benchmark |

#### Noise / Robustness Datasets

| Dataset | Description |
|:---|:---|
| CHiME-4/5/6/7/8 | Real noise environments; multi-mic; progressively harder |
| Aurora-4 | WSJ + 4 noise types + channel distortion |
| MUSAN | Music + speech + noise clips (addable library) |
| DNS Challenge | ICASSP headset + far-field industrial noise |
| DEMAND | 18 real-world noise environments at multiple SNRs |

#### Domain-Specific Datasets

| Dataset | Domain | Size |
|:---|:---|:---|
| SPGISpeech | Financial meetings | 5,000 h train / 100 h test |
| Earnings-22 | Earnings calls | 119 h total |
| Earnings-21 | Long-form earnings calls | 39 h, up to 60 min per file |
| Fisher Corpus | Telephone conversations | 1,960 h |
| SwitchBoard (eval2000) | Telephone spontaneous | 260 h |
| CallHome | Informal telephone, multilingual | 17 h |

#### Accent / Fairness Datasets

| Dataset | Coverage |
|:---|:---|
| CORAAL | Corpus of Regional African American Language |
| L2-ARCTIC | Arabic, Mandarin, Hindi, Korean, Spanish, Vietnamese |
| AESRC2020 | 8 accents: US, UK, Chinese, Indian, Japanese, Korean, Russian, Portuguese |
| CasualConversations-v1 | Age, gender, skin tone annotations for bias audits |

#### Additional Datasets

| Dataset | Coverage |
|:---|:---|
| Rev16 | 30 h podcasts, diverse speakers |
| Peoples Speech | 30K h open-source diverse speech |
| DIHARD | Hard diarization scenarios |

#### Dataset Selection Guide

| Use Case | Recommended Datasets |
|:---|:---|
| General English baseline | LibriSpeech (clean + other) |
| Accent robustness | Common Voice (filter by accent metadata) |
| Conversational/telephony | SwitchBoard, Fisher, Earnings-22 |
| Noisy environments | CHiME-4/5/6, DEMAND |
| Multilingual | FLEURS, VoxPopuli, MLS |
| Multi-speaker/meetings | AMI, DIHARD |
| Domain-specific | Build custom test sets from your production data |

---

### 2.6 Model Benchmark Scores

#### Open-Source Models — HuggingFace Open ASR Leaderboard (English ESB, 2024–2025)

*Hardware: NVIDIA A100-SXM4-80GB, CUDA 12.6, PyTorch 2.4.0. Greedy decoding, no external LM.*

| Model | Params | Mean WER | Lib-Clean | Lib-Other | SPGI | TED | VoxPop | GigaSpeech | E-22 | AMI | RTFx |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| NVIDIA Canary-Qwen-2.5B | 2.5B | **5.63%** | 1.60% | 3.10% | 1.90% | 2.72% | 5.66% | 9.41% | 10.42% | 10.18% | 418 |
| IBM Granite-Speech-3.3-8B | 8B | **5.74%** | 1.43% | 2.86% | 3.91% | 3.40% | 5.72% | 10.19% | 9.42% | 8.98% | 145 |
| NVIDIA Parakeet-TDT-0.6B-v2 | 600M | **6.05%** | 1.69% | 3.19% | 2.17% | 3.38% | 5.95% | 9.74% | 11.15% | 11.16% | 3,386 |
| NVIDIA Canary-1B | 1B | **6.50%** | 1.48% | 2.93% | 2.06% | 3.56% | 5.79% | 10.12% | 12.19% | 13.90% | 235 |
| NVIDIA Parakeet-TDT-1.1B | 1.1B | **7.02%** | 1.40% | 2.60% | 3.16% | 3.59% | 5.49% | 9.52% | 14.49% | 15.87% | 2,391 |
| Whisper large-v3 | 1.55B | **7.44%** | 1.8% | 3.6% | 2.94% | 3.86% | 9.54% | 10.02% | 11.29% | 15.95% | 146 |
| Distil-Whisper large-v3 | 756M | **7.52%** | 2.54% | 5.19% | 3.27% | 3.86% | 8.25% | 10.08% | 11.79% | 15.16% | 214 |
| Whisper large-v3-turbo | 809M | **7.83%** | 2.10% | 4.24% | 2.97% | 3.57% | 11.87% | 10.14% | 11.63% | 16.13% | 200 |
| Moonshine Base | 61M | **9.99%** | 3.38% | 8.15% | 5.46% | 5.65% | 10.84% | 12.08% | 16.85% | 17.49% | 566 |

> Accuracy leaders: LLM-decoder models (Canary-Qwen, IBM Granite). Speed leaders: CTC/TDT models (Parakeet) — 10–23× faster at ~1–2% WER penalty. AMI (meetings) and Earnings-22 are consistently hardest.

#### Open-Source Models — LibriSpeech WER Detail

| Model | Params | test-clean WER | test-other WER | RTF (GPU) |
|:---|:---|:---|:---|:---|
| Whisper large-v3 | 1.55B | 1.8% | 3.6% | ~0.3 |
| Whisper large-v3-turbo | 809M | 2.1% | 4.0% | ~0.05 |
| Whisper medium | 769M | 2.5% | 5.2% | ~0.15 |
| Whisper small | 244M | 3.4% | 7.6% | ~0.04 |
| Whisper base | 74M | 5.0% | 12.0% | ~0.02 |
| Canary-1B (NVIDIA) | 1B | 1.7% | 3.4% | ~0.08 |
| Conformer-CTC (NeMo) | 120M | 2.0% | 4.3% | ~0.03 |
| wav2vec 2.0 large | 317M | 1.8% | 3.3% | ~0.1 |

#### Distil-Whisper: Accuracy vs. Speed

| Model | Params | Speed vs. large-v3 | Short-Form WER | Sequential Long-Form | Chunked Long-Form |
|:---|:---|:---|:---|:---|:---|
| Whisper large-v3 | 1,550M | 1.0× | 8.4% | 10.0% | 11.0% |
| Distil-large-v3 | 756M | **6.3×** | 9.7% | 10.8% | 10.9% |
| Distil-large-v2 | 756M | 5.8× | 10.1% | 15.6% | 11.6% |

#### Commercial APIs — Batch WER (Picovoice Benchmark 2024)

| Engine | Lib-Clean | Lib-Other | TED-LIUM | Common Voice | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|
| Amazon Transcribe | 2.3% | 4.6% | 4.0% | 6.4% | **4.3%** |
| Azure Speech-to-Text | 2.9% | 6.0% | 4.6% | 8.4% | 5.5% |
| Whisper large-v3 (OSS) | 3.7% | 5.4% | 4.6% | 9.0% | 5.7% |
| Google Speech-to-Text | 5.3% | 10.5% | 5.5% | 14.3% | 8.9% |
| Picovoice Leopard | 5.1% | 11.1% | 6.4% | 16.1% | 9.7% |
| IBM Watson STT | 10.9% | 26.2% | 11.7% | 39.4% | 22.0% |

#### Commercial APIs — Streaming WER (Picovoice Benchmark 2024)

| Engine | Lib-Clean | Lib-Other | TED-LIUM | Common Voice | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|
| Amazon Transcribe Streaming | 2.6% | 5.5% | 4.8% | 9.4% | **5.6%** |
| Azure Speech Real-Time | 4.9% | 8.5% | 8.7% | 10.7% | 8.2% |
| Picovoice Cheetah | 5.4% | 11.4% | 6.4% | 17.0% | 10.1% |
| Moonshine Medium | 5.9% | 11.4% | 6.5% | 18.7% | 10.6% |
| Google STT Streaming | 8.6% | 14.3% | 7.9% | 16.8% | 11.9% |

#### Commercial APIs — Streaming Punctuation Error Rate

| Engine | Common Voice | FLEURS | VoxPopuli | Avg |
|:---|:---:|:---:|:---:|:---:|
| Azure Speech Real-Time | 5.6% | 17.6% | 25.9% | **16.4%** |
| Picovoice Cheetah | 6.5% | 14.4% | 27.4% | 16.1% |
| Amazon Transcribe Streaming | 13.2% | 24.4% | 35.5% | 24.4% |
| Google STT Streaming | 20.2% | 42.7% | 45.0% | 36.0% |

#### Commercial APIs — Full Feature Comparison

| Provider | Model | English WER (clean) | Streaming Latency | Strength |
|:---|:---|:---|:---|:---|
| **Deepgram** | Nova-3 | ~4–6% | ~200 ms | Ultra-low latency, real-time, voice agents |
| **OpenAI** | Whisper API | ~2–3% | N/A (batch) | Highest accuracy (batch) |
| **AssemblyAI** | Universal-2 | ~3–5% | ~300 ms | Entity recognition, formatting |
| **Google** | Chirp 2 | ~3–5% | ~200 ms | 100+ languages, ecosystem |
| **Azure** | Speech (Whisper-based) | ~3–5% | ~250 ms | Enterprise, custom models |
| **AWS** | Transcribe | ~5–8% | ~500 ms | HIPAA, PII redaction |
| **Speechmatics** | Ursa 3 | ~4–6% | ~200 ms | Accent robustness, diarization |

> Vendor-reported WER uses favorable test conditions. Always benchmark on your own domain data for production decisions.

#### Conversational / Noisy Performance

| Provider / Model | SwitchBoard WER | Noisy (10 dB SNR) | Accented English |
|:---|:---:|:---:|:---:|
| Whisper large-v3 | ~8–12% | ~10–18% | ~5–12% |
| Deepgram Nova-3 | ~8–11% | ~8–15% | ~6–10% |
| AssemblyAI Universal-2 | ~7–10% | ~10–16% | ~6–11% |
| Google Chirp 2 | ~9–13% | ~12–20% | ~6–12% |

#### Multilingual WER — Whisper vs. Amazon Transcribe

| Language | Whisper large-v3 | Amazon Transcribe |
|:---|:---:|:---:|
| French | 8.3% | 6.3% |
| German | 7.4% | 7.6% |
| Spanish | 5.5% | 5.3% |
| Portuguese | 5.7% | 6.6% |

#### WER Expectations by Scenario (2025)

| Scenario | SOTA WER | Typical Production WER |
|:---|:---:|:---:|
| Audiobook narrated (LibriSpeech clean) | ~1.4% | 2–5% |
| TED-style prepared speech | ~2.7% | 4–8% |
| Financial earnings (SPGISpeech) | ~1.9% | 5–10% |
| Spontaneous meeting (AMI) | ~9% | 15–25% |
| Financial spontaneous (Earnings-22) | ~9.4% | 12–20% |
| Noisy speech (SNR 0 dB) | ~10–12% | 20–40% |

*Real-world production WER is typically 3–5× higher than clean academic benchmarks.*

---

### 2.7 Leaderboards

| Leaderboard | URL | Scope | Metrics |
|:---|:---|:---|:---|
| **HF Open ASR Leaderboard v2** | huggingface.co/spaces/hf-audio/open_asr_leaderboard | 60+ models, 11 datasets | WER, RTFx |
| **Picovoice STT Benchmark** | github.com/Picovoice/speech-to-text-benchmark | Commercial APIs + OSS; batch + streaming | WER, latency, punctuation |
| **ESB Leaderboard** | huggingface.co/spaces/esb/leaderboard | 8 English datasets | WER |
| **SUPERB** | superbbenchmark.org | 10 tasks across pre-trained representations | Task-specific |
| **Papers With Code — LibriSpeech** | paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean | Full research history | WER |
| **MLCommons MLPerf Inference** | mlcommons.org | Standardized inference speed | Latency, throughput |
| **Artificial Analysis** | artificialanalysis.ai | Commercial API speed and cost | WER, latency, cost |
| **NIST OSER / CHiME workshops** | nist.gov/itl/iad/mig | Government/DARPA; blind test sets | WER, DER |

---

### 2.8 Best Practices

#### Text Normalization

Always normalize both reference and hypothesis before computing WER with the same pipeline:

```python
import jiwer

transform = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])

wer = jiwer.wer(reference, hypothesis,
                truth_transform=transform,
                hypothesis_transform=transform)
```

OpenAI's English normalizer also: expands contractions, normalizes numbers to word form.

#### Statistical Significance

- **Bootstrap resampling** (N = 1,000) over utterances to compute 95% CI on WER
- **MAPSSWE test** (Matched-Pairs Sentence Segment WER) for pairwise model comparison
- Minimum test set size: **~10 hours** for reliable benchmarking; **~1 hour** for quick screening
- Only claim one model beats another if p < 0.05

#### Evaluation Checklist

```
- [ ] Normalize text (lowercase, strip punctuation, expand numbers) with same pipeline for ref + hyp
- [ ] Test on clean AND noisy data
- [ ] Include domain-specific vocabulary in test sets
- [ ] Measure WER per accent/dialect group separately (report fairness deltas)
- [ ] Report RTFx alongside WER for real-time use cases
- [ ] Test streaming vs. batch modes separately
- [ ] Measure endpointing / finalization latency
- [ ] Evaluate confidence score calibration (ECE, reliability diagram)
- [ ] Evaluate with your production audio (not just benchmarks)
- [ ] Track WER over time to detect regression
- [ ] Report confidence intervals (bootstrap or per-utterance)
- [ ] Use MAPSSWE test for pairwise model comparisons
```

---

## 3. TTS / Speech Synthesis Evaluation

### 3.1 Automated Metrics

#### MOS (Mean Opinion Score)

The gold standard for perceived quality. Human listeners rate each sample 1–5.

```
MOS = (1/N) × Σ rᵢ    where rᵢ ∈ {1, 2, 3, 4, 5}

95% CI: MOS ± 1.96 × (σ / √N)
```

| MOS | Quality |
|:---|:---|
| 4.5+ | Excellent — indistinguishable from natural speech |
| 4.0–4.5 | Good — just perceptibly different |
| 3.0–4.0 | Fair — perceptible but not annoying |
| 2.0–3.0 | Poor — annoying impairments |
| < 2.0 | Bad — very annoying, communication failure |

Reference points: Natural human speech ~4.5, PSTN telephony ~4.0, narrowband codec ~3.5.

**MOS variants:**
- **N-MOS** — naturalness only
- **Q-MOS** — overall quality
- **S-MOS** (SMOS) — speaker similarity (for voice cloning)
- **MOS-E** — emotion expressiveness

> MOS scores are not comparable across papers without shared anchors, test sets, and listener pool. Always report 95% CI.

#### UTMOS (UTokyo-SaruLab MOS)

Automated neural MOS predictor (no humans required).

**How it works:** wav2vec 2.0 or HuBERT backbone extracts speech representations → regression head predicts scalar MOS in [1.0, 5.0] → optional listener conditioning module models rater-specific biases.

System-level Pearson correlation with human MOS: **~0.95** on VoiceMOS Challenge 2022/2023.

```python
pip install utmos
```
```python
from utmos import UTMOSScore
scorer = UTMOSScore()
score = scorer.score("path/to/audio.wav")  # Returns float ~[1.0, 5.0]
```

> Well-calibrated for English; degrades on low-resource languages. Use as a development-time filter, not a human MOS substitute.

#### DNSMOS (Microsoft Deep Noise Suppression MOS)

Calibrated against ITU-T P.835. Better than UTMOS at detecting codec artifacts.

| Score | Measures |
|:---|:---|
| DNSMOS OVRL | Overall speech quality |
| DNSMOS SIG | Signal quality (clarity of speech) |
| DNSMOS BAK | Background quality (absence of noise/artifacts) |

```python
from DNSMOS import dnsmos_local
results = dnsmos_local.compute_dnsmos("audio.wav")
# Returns {'OVRL': float, 'SIG': float, 'BAK': float}
```

#### NISQA (Non-Intrusive Speech Quality Assessment)

CNN + attention architecture predicting quality for TTS and telephony. Range: 1–5. Complementary to UTMOS — better suited for degraded/telephony conditions.

#### WER via ASR Re-synthesis (Intelligibility)

Synthesize audio → pass through ASR → compare transcript to original text.

```
WER = (S + D + I) / N
```

Standard ASR for cross-paper comparability: **Whisper large-v3** or wav2vec 2.0 (fine-tuned on LibriSpeech 960h).

Additional intelligibility targets:
- **CER** (Character Error Rate): target < 1%
- **Phoneme Error Rate**: target < 5%

```python
import whisper
import jiwer

model = whisper.load_model("large-v3")
result = model.transcribe("synthesized_audio.wav")
hypothesis = result["text"]
reference = "The original input text sent to TTS."

wer = jiwer.wer(
    jiwer.RemovePunctuation()(reference.lower()),
    jiwer.RemovePunctuation()(hypothesis.lower())
)
print(f"Round-trip WER: {wer:.2%}")
```

#### PESQ (Perceptual Evaluation of Speech Quality)

ITU-T P.862 / P.862.2 standard. **Requires a paired clean reference signal.**

| Mode | Sample Rate | Range |
|:---|:---|:---|
| Narrowband (P.862) | 8 kHz | −0.5 to 4.5 |
| Wideband (P.862.2) | 16 kHz | −0.5 to 4.5 → MOS-LQO |

POLQA (ITU-T P.863) is the modern successor. Best used for vocoder degradation measurement (comparing vocoder output vs. ground-truth recording). Not applicable for pure text-to-speech evaluation when no reference audio exists.

```python
from pesq import pesq
score = pesq(16000, reference_signal_array, degraded_signal_array, 'wb')
```

#### STOI (Short-Time Objective Intelligibility)

Measures intelligibility via temporal envelope correlation across 1/3-octave bands. Range: [0, 1].

| STOI | Intelligibility |
|:---|:---|
| > 0.90 | High |
| 0.70–0.90 | Moderate |
| < 0.70 | Poor |

```python
from pystoi import stoi
score = stoi(reference_array, degraded_array, sample_rate, extended=True)
# extended=True uses ESTOI — better for non-stationary noise
```

#### MCD (Mel Cepstral Distortion)

Compares MFCC sequences between synthesized and reference speech after DTW alignment.

```
MCD = (10 / ln(10)) × √(2 × Σ (mcep_ref(k) − mcep_syn(k))²)   [in dB]

Summed over k = 1 to K MFCC dimensions (typically K = 13 or K = 25)
Requires DTW alignment before comparison
```

| MCD (dB) | Quality |
|:---|:---|
| < 4 | Excellent |
| 4–6 | Good |
| 6–8 | Acceptable |
| > 8 | Poor |

```python
import librosa
import numpy as np
from dtw import dtw

def compute_mcd(ref_path, syn_path, n_mfcc=13):
    ref, sr = librosa.load(ref_path, sr=22050)
    syn, _  = librosa.load(syn_path, sr=22050)
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc).T
    syn_mfcc = librosa.feature.mfcc(y=syn, sr=sr, n_mfcc=n_mfcc).T
    alignment = dtw(ref_mfcc, syn_mfcc, dist_method='euclidean')
    mcd = (10.0 * np.sqrt(2) / np.log(10)) * alignment.distance / len(ref_mfcc)
    return mcd
```

#### SSIM on Mel Spectrograms

Image structural similarity metric adapted to compare mel spectrograms as 2D images.

```
SSIM(x, y) = [(2μₓμᵧ + c₁)(2σₓᵧ + c₂)] / [(μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂)]
```

Range: [0, 1]. Best used as a low-level reconstruction sanity check (e.g., vocoder quality). MCD is generally a better perceptual metric for speech.

#### Speaker Similarity (Cosine Embedding Distance)

Critical for voice cloning and personalization.

```
SpeakerSim = cos(e_ref, e_syn) = (e_ref · e_syn) / (‖e_ref‖ × ‖e_syn‖)
```

| Score | Interpretation |
|:---|:---|
| > 0.90 | Very high — near-perfect cloning |
| 0.80–0.90 | Good cloning quality |
| 0.70–0.80 | Partial — some characteristics captured |
| < 0.70 | Poor cloning |

```python
# SpeechBrain ECAPA-TDNN
from speechbrain.pretrained import SpeakerRecognition
verifier = SpeakerRecognition.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
score, prediction = verifier.verify_files("reference.wav", "synthesized.wav")

# Resemblyzer (faster, lower accuracy)
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
encoder = VoiceEncoder()
embed_ref = encoder.embed_utterance(preprocess_wav("reference.wav"))
embed_syn = encoder.embed_utterance(preprocess_wav("synthesized.wav"))
similarity = np.dot(embed_ref, embed_syn) / (
    np.linalg.norm(embed_ref) * np.linalg.norm(embed_syn)
)
print(f"Speaker Similarity: {similarity:.4f}")  # > 0.85 = good clone
```

#### TTSDS2 (Text-to-Speech Distribution Score 2)

Evaluates multiple perceptual dimensions (speaker identity, intelligibility, prosody) jointly. Strong correlation with human ratings across diverse TTS systems.

#### VERSA Toolkit

Open-source toolkit consolidating **90+ metrics** (MOS prediction, speaker similarity, MCD, F0, PESQ, STOI, and more). GitHub: `github.com/shinjiwlab/versa`

```python
pip install speechmetrics   # Simpler alternative: PESQ, STOI, MOSNet
```

#### Prosody Metrics

**F0 (Pitch) RMSE and Correlation** — compare fundamental frequency contours (voiced frames only):
```
F0_RMSE = √[ (1/T) × Σ (F0_syn(t) − F0_ref(t))² ]
```
Also compute Pearson/Spearman correlation of F0 contours. Tool: `pyworld`.

**Phoneme Duration RMSE** — via Montreal Forced Aligner (MFA):
```
Duration_RMSE = √[ (1/N) × Σ (d_syn(i) − d_ref(i))² ]
```

**Energy/Intensity Contour RMSE** — captures stress and emphasis:
```
Energy_RMSE = √[ (1/T) × Σ (E_syn(t) − E_ref(t))² ]
```

**nPVI (normalized Pairwise Variability Index)** — measures rhythmic variability across successive syllables. High nPVI = stress-timed (English), low nPVI = syllable-timed (French, Spanish).

#### Speaking Rate

Natural English narration: **130–180 WPM** (~3.3–5.9 syllables/second).

```python
import librosa
duration = librosa.get_duration(path="audio.wav")
wpm = (len(text.split()) / duration) * 60
```

Target: ±15% of expected rate for the domain (narration vs. conversational vs. news reading).

#### Streaming Latency (TTFB / RTF)

| Metric | Definition | Target |
|:---|:---|:---|
| **TTFB** (Time to First Byte) | API call to first audio byte received | < 250 ms conversational |
| **RTF** | `synthesis_time / audio_duration` | < 1.0 streaming; < 0.1 batch |
| **p99 Latency** | 99th percentile TTFB across requests | < 500 ms under load |

#### Emotion Classification Accuracy

Synthesize utterances labeled with target emotions → pass through SER (Speech Emotion Recognition) classifier (wav2vec 2.0 fine-tuned on IEMOCAP) → report accuracy + per-class F1.

Ekman's Big-6 + neutral: happy, sad, angry, neutral, fearful, surprised, disgusted.
Target: **> 75% classification accuracy** for expressive TTS systems.

#### Voice Consistency (Long-Form)

1. Synthesize N segments from the same long-form text
2. Extract speaker embeddings per segment
3. Measure inter-segment cosine similarity variance — low variance = consistent voice
4. Track F0 mean and standard deviation drift across segments

---

### 3.2 Human Evaluation Protocols

#### MOS (ITU-T P.800)

**Standard protocol:**
- 20–30 native listener raters per condition
- Absolute Category Rating (ACR): each sample rated independently 1–5
- Randomized sample presentation; maximum 30-minute sessions to prevent fatigue
- Include a known-bad anchor and natural speech reference to calibrate raters
- Recruit via Prolific Academic (higher quality), MTurk (large-scale), or in-house panels

Report: `MOS ± 1.96 × (σ / √N)`. A difference is meaningful only if CI ranges do not overlap.

#### MUSHRA (ITU-R BS.1534)

Preferred when comparing multiple high-quality systems (when all score > 4.0 MOS and absolute ratings cluster too tightly to discriminate).

**Protocol:**
- Listeners hear all versions of the same utterance simultaneously
- Rate each on a **0–100 continuous scale**
- Hidden reference (natural speech) always present — raters who score it < 90 are filtered as unreliable
- Low-quality anchor (e.g., narrowband 3.5 kHz low-pass filtered) establishes floor at 0–20
- Minimum 15 raters (typically 20–30)

| MUSHRA Score | Quality |
|:---|:---|
| 80–100 | Excellent |
| 60–80 | Good |
| 40–60 | Fair |
| 20–40 | Poor |
| 0–20 | Bad |

Tool: **WebMUSHRA** — open-source browser-based implementation: `github.com/audiolabs/webMUSHRA`

#### ABX Test

Three stimuli: A (system 1), B (system 2), X (one of A or B). Listener identifies whether X is A or B. If accuracy ≤ 55% (near-chance), systems are perceptually indistinguishable for the tested attribute.

#### CMOS (Comparative MOS)

Rate relative quality difference on a **−3 to +3 scale** against a reference condition. Used for comparative assessment when the absolute MOS anchor is fixed.

#### Preference Test (A/B Forced Choice)

Listeners hear two utterances and select which they prefer. Report **preference rate** with 95% binomial CI. No significant preference if p > 0.05 (chi-squared or binomial test against 50%).

#### SMOS (Speaker MOS)

Same structure as standard MOS but for voice cloning: raters hear a reference recording of the target speaker alongside a synthesized sample, then rate perceptual similarity 1–5.

#### Crowdsourced Evaluation

| Platform | Notes |
|:---|:---|
| **Prolific Academic** | Higher-quality rater pool; recommended for research |
| **Amazon Mechanical Turk** | Large-scale MOS collection; requires quality controls |
| **TTS Arena (HF)** | Community Elo-based blind voting; public leaderboard |

---

### 3.3 Evaluation Dimensions

| Dimension | Metrics | Method |
|:---|:---|:---|
| **Naturalness** | MOS, UTMOS, DNSMOS, MUSHRA | Human ratings + neural predictors |
| **Intelligibility** | WER/CER (ASR round-trip), STOI | ASR re-synthesis + signal analysis |
| **Prosody — Rhythm** | Duration RMSE, nPVI | DTW-aligned comparison to reference |
| **Prosody — Intonation** | F0 RMSE, F0 correlation | pyworld extraction on voiced frames |
| **Prosody — Stress** | F0 peak alignment, energy contour | Frame-level comparison |
| **Speaker Identity** | SpeakerSim (cosine), SMOS | Speaker embedding comparison |
| **Emotion / Expressiveness** | SER classifier accuracy + F1, MOS-E | Classifier on synthesized speech + human eval |
| **Audio Signal Quality** | DNSMOS, PESQ, STOI | Neural + signal-based predictors |
| **Speaking Rate** | WPM calculation | `len(words) / duration × 60` |
| **Streaming Latency** | TTFB p50/p90/p99, RTF | Timing instrumentation |
| **Long-form Consistency** | Inter-segment SpeakerSim variance, F0 drift | Compare embeddings across segments |
| **Multilingual Quality** | Per-language WER + MOS | Language-specific evaluations |
| **Robustness** | Hallucination rate, repetition rate | Test on edge-case inputs: numbers, URLs, abbreviations, heavy punctuation |

---

### 3.4 Benchmark Datasets

| Dataset | Speakers | Duration | Sample Rate | Language | Primary Use |
|:---|:---|:---|:---|:---|:---|
| **LJSpeech** | 1 (female, American English) | 24 h | 22,050 Hz | English | Universal single-speaker baseline |
| **VCTK** | 110 (mixed accent) | 44 h | 48,000 Hz | English | Multi-speaker, voice cloning, accent robustness |
| **LibriTTS** | 2,456 | 585 h | 24,000 Hz | English | Large-scale multi-speaker |
| **LibriTTS-R** | 2,456 | 585 h | 24,000 Hz | English | Restored/higher-quality version of LibriTTS |
| **CMU ARCTIC** | 7 | ~8 h | 16,000 Hz | English | Phonetically balanced |
| **Hi-Fi TTS** | 10 | 291 h | 44,100 Hz | English | High-fidelity vocoder evaluation |
| **EXPRESSO** | 4 | 40 h | 44,100 Hz | English | Emotion and style evaluation (26 styles) |
| **Blizzard Challenge** | varies/year | varies | varies | English | Annual challenge; year-over-year comparison |
| **EmoV-DB** | 4 | ~7 h | 44,100 Hz | English | Emotion TTS (5 classes) |
| **EARS** | 107 | varies | 44,100 Hz | English | 24–26 speaking styles + emotions |
| **CSS10** | 1/language | ~112 h total | 22,050 Hz | 10 languages | Single-speaker multilingual TTS |
| **MLS** | thousands | 50,000+ h | 16,000 Hz | 8 languages | Multilingual multi-speaker |
| **CommonVoice v15** | tens of thousands | 30,000+ h | 16,000 Hz | 120+ languages | Multilingual generalization |

> **LJSpeech caveat:** "Human parity" claimed on LJSpeech does not generalize to conversational, emotional, or multi-speaker speech — it is a single-speaker read-speech corpus.

---

### 3.5 Model Benchmark Scores

> **Critical context:** MOS figures are only comparable under identical conditions (same test sentences, listener pool, anchor stimuli, rating interface). Numbers below aggregate from published papers and independent evaluations.

#### Open-Source / Research Models

| Model | Year | Architecture | MOS (en) | WER (%) | SpeakerSim | Notes |
|:---|:---|:---|:---|:---|:---|:---|
| Tacotron 2 + WaveNet | 2018 | Autoregressive Seq2Seq | ~4.5 | ~2–4% | N/A | Established neural TTS baseline |
| FastSpeech 2 + HiFi-GAN | 2020 | Non-autoregressive | ~4.1–4.3 | ~2–3% | N/A | 50× faster; slightly lower quality |
| VITS | 2021 | Normalizing flow, end-to-end | ~4.43 | ~1.5–2% | N/A | Best quality/speed tradeoff at release |
| NaturalSpeech | 2022 | Diffusion + VAE | ~4.91 | ~1.2% | N/A | First to claim human parity on LJSpeech |
| NaturalSpeech 3 | 2024 | Factorized codec + diffusion | ~4.7 | ~0.8% | ~0.91 | Multi-speaker SOTA (Microsoft 2024) |
| VALL-E | 2023 | LM over EnCodec tokens | ~4.7 | ~1.5–3% | ~0.93 | Zero-shot 3-second cloning |
| VALL-E 2 | 2024 | VALL-E + codec merging | ~4.8 | ~1.0% | ~0.94 | Claimed human parity on VCTK/LibriSpeech |
| VoiceBox | 2023 | Flow Matching | ~4.57 | ~1.9% | ~0.81 | Meta; first large-scale flow-matching TTS |
| StyleTTS 2 | 2023 | Style diffusion | ~4.47 | ~2.0% | N/A | Strong open-source single-speaker |
| XTTS v2 (Coqui) | 2023 | Cross-lingual zero-shot | ~4.0–4.3 | ~3–5% | ~0.75–0.85 | 17 languages; strong open-source community |
| Bark (Suno AI) | 2023 | GPT-like generative | ~3.8–4.1 | ~5–10% | ~0.65–0.75 | Highly expressive; unreliable intelligibility |
| VoiceCraft | 2024 | Token-based in-context | ~4.2–4.4 | ~2.5–4% | ~0.80 | Designed for natural speech editing |
| F5-TTS | 2024 | Flow Matching + Conformer | ~4.1–4.5 | ~1.5–3% | ~0.82–0.88 | Strong zero-shot; fast inference |
| Kokoro | 2024 | StyleTTS2-based | ~4.1–4.3 | ~1.5–2.5% | N/A | 82M params; highly efficient; top TTS Arena OSS |
| Parler-TTS | 2024 | Prompt-based voice description | ~3.8–4.0 | ~2–4% | N/A | 880M params; Apache 2.0 |
| MetaVoice | 2024 | Zero-shot cloning focus | ~3.7–3.9 | ~2–4% | N/A | Apache 2.0 |
| Chatterbox (Resemble AI) | 2025 | Open-source zero-shot | ~4.2–4.5 | ~2–3% | ~0.82–0.90 | First open-source competitive to commercial |
| Orpheus TTS | 2025 | LLaMA-based LM | ~4.4 | ~2% | ~0.82 | Strong emotion; Canopy Labs |

#### Commercial Systems

| System | MOS Estimate | WER Estimate | TTFB | Notable Strengths |
|:---|:---|:---|:---|:---|
| Inworld TTS-1.5-Max | ~4.3–4.5 | ~1.5–2% | ~150 ms | TTS Arena #1 (April 2026); gaming/agents |
| ElevenLabs v3 (Turbo) | ~4.5–4.7 | ~1–2% | ~200 ms | Best naturalness + cloning + emotion range |
| ElevenLabs Multilingual v2 | ~4.4–4.6 | ~1.5–3% | ~200 ms | 30+ languages; top multilingual |
| OpenAI TTS-1-HD | ~4.4–4.6 | ~1–2% | ~300 ms | Very natural prosody |
| OpenAI TTS-1 | ~4.1–4.3 | ~1.5–2.5% | ~150 ms | Lower latency; good for real-time |
| Google TTS Neural2 | ~4.3–4.5 | ~1–2% | ~100–200 ms | Strong prosody; good for long-form |
| Google TTS Studio | ~4.4–4.6 | ~1–2% | ~150–250 ms | Top-tier Google voices |
| Azure TTS Neural | ~4.2–4.5 | ~1–2% | ~100–250 ms | Rich SSML support; wide language coverage |
| Azure TTS Custom Neural | ~4.4–4.6 | ~1–2% | ~100–250 ms | Excellent for trained brand voices |
| Amazon Polly Neural | ~4.0–4.2 | ~1.5–3% | ~100–200 ms | Cost-effective; brand-safe |
| Deepgram Aura | ~4.0–4.3 | ~2–3% | 50–100 ms | Ultra-low latency; built for voice agents |
| Cartesia Sonic 3 | ~4.2–4.5 | ~1.5–2.5% | ~75 ms | Ultra-low latency; SSM architecture |
| PlayHT PlayDialog | ~4.3–4.5 | ~1.5–2.5% | ~150–300 ms | Conversational style; emotional expression |
| Fish Audio S2 Pro | ~4.0–4.2 | ~2–3% | ~150 ms | Strong price/performance ratio |
| Rime TTS | ~4.3–4.5 | ~1.5–2% | ~100–200 ms | Natural conversational quality |
| Resemble AI | ~4.3–4.5 | ~1–2% | ~100–250 ms | Strong voice cloning; emotion control |

---

### 3.6 Leaderboards

#### TTS Arena (Hugging Face / TTS-AGI)

**URL:** `huggingface.co/spaces/TTS-AGI/TTS-Arena-V2`

Elo-based pairwise preference voting by crowdsourced human listeners (same model as LMSYS Chatbot Arena). Most widely cited public TTS leaderboard.

**Approximate Elo rankings (April 2026):**

| Rank | System | Elo (approx.) | Category |
|:---|:---|:---|:---|
| 1 | Inworld TTS-1.5-Max | ~1,200+ | Commercial |
| 2 | ElevenLabs v3 | ~1,180+ | Commercial |
| 3 | Kokoro | ~1,150+ | Open-source |
| 4 | Cartesia Sonic 3 | ~1,130+ | Commercial |
| 5 | Fish Audio S2 Pro | ~1,110+ | Commercial |
| 6 | OpenAI TTS-1-HD | ~1,090+ | Commercial |
| 7 | XTTS v2 | ~1,050+ | Open-source |

> Rankings fluctuate continuously — check the live leaderboard for current standings. Voter pool is self-selected (tech-savvy, English-dominant).

#### Blizzard Challenge (2005–2023)

`synsig.org/index.php/Blizzard_Challenge`

Annual speech synthesis challenge with professional MOS evaluations and a fixed listener panel. The most rigorous historical benchmark — enables true year-over-year progress tracking across 100+ systems over 18 years.

#### VoiceMOS Challenge (2022, 2023)

`voicemos-challenge-2023.github.io`

Provides human MOS labels for hundreds of TTS and voice conversion systems. This is the training and validation data behind UTMOS.

#### SUPERB (Speech processing Universal PERformance Benchmark)

`superbbenchmark.org`

Not a TTS benchmark per se, but the speaker verification and emotion recognition models used to evaluate TTS output are ranked here.

#### Papers With Code — Speech Synthesis

`paperswithcode.com/task/speech-synthesis`

Tracks published SOTA MOS scores on LJSpeech, VCTK, and LibriTTS across academic papers.

#### SpeechIO TIOBE (Chinese TTS)

39 test sets across news, novels, technical text, poetry, and regional dialects. Most comprehensive benchmark for Chinese-language TTS.

#### Artificial Analysis

`artificialanalysis.ai`

Tracks commercial TTS API quality and cost/latency trade-offs. Updated regularly.

---

### 3.7 Consolidated Thresholds Table

| Metric | Excellent | Good | Acceptable | Poor |
|:---|:---|:---|:---|:---|
| MOS / UTMOS | > 4.3 | 3.8–4.3 | 3.3–3.8 | < 3.0 |
| Speaker Similarity | > 0.90 | 0.80–0.90 | 0.70–0.80 | < 0.70 |
| Round-trip WER | < 2% | 2–5% | 5–10% | > 10% |
| MCD | < 4.0 dB | 4.0–6.0 dB | 6.0–8.0 dB | > 8.0 dB |
| F0 RMSE | < 10 Hz | 10–20 Hz | 20–40 Hz | > 40 Hz |
| PESQ | > 3.5 | 2.5–3.5 | 1.5–2.5 | < 1.5 |
| STOI | > 0.90 | 0.80–0.90 | 0.65–0.80 | < 0.65 |
| TTS TTFB | < 100 ms | 100–200 ms | 200–400 ms | > 400 ms |

---

### 3.8 End-to-End Evaluation Pipeline

```
Phase 1 — Prepare test sets:
  - 100 phonetically balanced sentences (Harvard Sentences / VCTK prompts)
  - 10 × 200–500-word passages for long-form
  - 50 technical sentences (numbers, abbreviations, proper nouns)
  - 30 sentences × 7 emotion categories (for expressive TTS)
  - 20 edge-case sentences (URLs, currencies, dates, heavy punctuation)
  - For voice cloning: include 3s, 10s, 30s reference prompts

Phase 2 — Synthesize and collect audio

Phase 3 — Automated metrics (no human required):
  - Intelligibility:      Whisper large-v3 WER / CER
  - Naturalness:          UTMOS
  - Audio quality:        DNSMOS (OVRL, SIG, BAK)
  - Speaker similarity:   ECAPA-TDNN cosine similarity
  - Prosody:              F0 RMSE, Duration RMSE (pyworld + MFA)
  - Speaking rate:        syllables/second
  - Latency:              TTFB p50/p90/p99 + RTF

Phase 4 — Human evaluation (MUSHRA):
  - 30 representative sentences
  - 20+ native speaker raters (Prolific Academic or in-house)
  - Run via WebMUSHRA
  - Report mean MUSHRA ± 95% CI per condition

Phase 5 — Reporting:
  - Summary table: all metrics per system
  - Category breakdown: short / long-form / emotional / technical
  - Failure analysis: most common ASR error types, emotion misclassifications
  - Latency profile: TTFB distribution (p50/p90/p99)

Phase 6 — Statistical significance:
  - Bootstrap CI or Wilcoxon signed-rank test for human scores
  - Minimum p < 0.05 to claim one system beats another

Phase 7 — Long-form consistency check:
  - Inter-segment SpeakerSim variance
  - F0 drift across 5+ minute outputs
```

---

### 3.9 Tooling

| Tool | Metrics | Usage |
|:---|:---|:---|
| **VERSA** | 90+ metrics (MOS prediction, similarity, MCD, F0, PESQ, STOI) | `github.com/shinjiwlab/versa` |
| **speechmetrics** | PESQ, STOI, MOSNet | `pip install speechmetrics` |
| **pesq** | PESQ (P.862) | `pip install pesq` |
| **pystoi** | STOI, ESTOI | `pip install pystoi` |
| **UTMOS** | Neural MOS prediction | `pip install utmos` |
| **jiwer** | WER/CER for ASR round-trip | `pip install jiwer` |
| **resemblyzer** | Speaker embeddings + cosine similarity | `pip install resemblyzer` |
| **SpeechBrain** | ECAPA-TDNN speaker verification | `pip install speechbrain` |
| **pyworld** | F0 extraction for F0 RMSE | `pip install pyworld` |
| **librosa** | Mel spectrograms, MFCCs, audio analysis | `pip install librosa` |
| **WebMUSHRA** | Browser-based MUSHRA listening tests | `github.com/audiolabs/webMUSHRA` |

---

### 3.10 Evaluation Checklist

```
- [ ] Measure UTMOS / DNSMOS for automated naturalness scoring
- [ ] Compute WER via ASR round-trip (Whisper large-v3)
- [ ] Compute speaker similarity (cosine on ECAPA-TDNN / WavLM embeddings)
- [ ] Measure MCD and F0 RMSE against reference (if reference audio available)
- [ ] Run human MOS or MUSHRA evaluation (20+ raters, ITU-T P.800 / BS.1534)
- [ ] Test on diverse inputs (numbers, URLs, acronyms, long-form)
- [ ] Measure TTFB and RTF for latency-critical applications
- [ ] Evaluate emotion / expressiveness if applicable (SER classifier)
- [ ] Test long-form consistency (5+ minute passages, SpeakerSim variance)
- [ ] Check hallucination rate (extra/missing words in synthesis)
- [ ] Compare against TTS Arena and Artificial Analysis rankings
- [ ] Benchmark on target language/accent if multilingual
```

---

## Appendix

### A. Pipeline Metric Selection Quick Reference

| Goal | Metric | Requires reference? |
|:---|:---|:---|
| Per-stage latency | VAD/STT/EOU/LLM/TTS timestamps | No |
| End-to-end response time | e2e_latency | No |
| Mouth-to-ear latency | Stereo waveform analysis | No |
| Turn-taking quality | Turn Gap, Double-Talk Ratio, TPM | No |
| Interruption accuracy | Barge-in F1, Backchanneling Rejection | No |
| Business outcomes | TSR, FCR, Containment Rate | No |
| Cost per session | cost_stt + cost_llm + cost_tts + cost_infra | No |
| Scalability | P99 < 3× P50, Session Stability > 99.5% | No |

### B. STT Metric Selection Quick Reference

| Goal | Metric | Requires reference? |
|:---|:---|:---|
| Overall word accuracy | WER | Yes (text) |
| Character-level (logographic) | CER | Yes (text) |
| Over/under-generation combined | MER, WIL | Yes (text) |
| Utterance-level reliability | SER | Yes (text) |
| Domain term accuracy | NE-WER, KRR | Yes (text + NER) |
| Punctuation output quality | Punctuation F1 | Yes (text) |
| Real-time capability | RTFx | No |
| Streaming responsiveness | TTFW, Partial Result Latency | No |
| Speaker assignment accuracy | DER, JER, SA-WER | Yes (timestamps) |
| Confidence quality | ECE, AUROC | Yes (labels) |
| Semantic faithfulness | LLM-as-Judge, SeMaScore | Yes (text) |
| Intent preservation | Intent Preservation score | Yes (intent labels) |

### C. TTS Metric Selection Quick Reference

| Goal | Metric | Requires reference audio? |
|:---|:---|:---|
| Overall naturalness (human) | MOS, MUSHRA | No |
| Naturalness (automated, fast) | UTMOS, DNSMOS | No |
| Intelligibility | WER / CER (ASR-based) | No |
| Audio signal fidelity | PESQ, STOI | Yes |
| Spectral reconstruction quality | MCD, SSIM on mel | Yes |
| Speaker identity preservation | SpeakerSim, SMOS | Yes (speaker reference) |
| Pitch / intonation accuracy | F0 RMSE, F0 correlation | Yes |
| Rhythm accuracy | Duration RMSE, nPVI | Yes |
| Emotion expressiveness | SER classifier accuracy + F1 | No |
| Streaming latency | TTFB p50/p90/p99, RTF | No |
| Long-form voice consistency | Inter-segment SpeakerSim variance | No |
| Relative multi-system ranking | MUSHRA, ABX, Elo (TTS Arena) | No |

### D. Key Principles

| Principle | Detail |
|:---|:---|
| No single metric is sufficient | Combine: UTMOS + WER + SpeakerSim + human MUSHRA at minimum for TTS; WER + RTFx + domain-specific test set for STT |
| MOS numbers are not portable | Raw MOS is meaningless without shared anchors, test sets, and listener pool — always report 95% CI |
| UTMOS is a proxy | Use it for fast iteration; run a proper human study for final decisions |
| TTFB is underrated | For voice agents, 500 ms latency kills UX regardless of quality score |
| MUSHRA beats MOS for comparisons | When all systems score > 4.0 MOS, relative MUSHRA is far more discriminative |
| LJSpeech "human parity" is narrow | Does not imply parity on conversational, multilingual, or emotionally expressive synthesis |
| Fairness deltas matter | Always report per-accent and per-demographic WER gaps, not just averages |
| Leaderboards lag | Always run your own evaluation for production decisions; TTS Arena Elo shifts continuously |

---

## Sources

- [LiveKit Agents Data Hooks & Metrics Reference](https://docs.livekit.io/deploy/observability/data/)
- [LiveKit Turn Detector Benchmarks](https://docs.livekit.io/agents/logic/turns/turn-detector/)
- [LiveKit Noise Cancellation WER Benchmarks](https://docs.livekit.io/transport/media/noise-cancellation/)
- [LiveKit Self-Hosted Load Test Results](https://docs.livekit.io/deploy/custom/deployments/)
- [LiveKit Adaptive Interruption Handling](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/)
- [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [Picovoice Speech-to-Text Benchmark](https://github.com/Picovoice/speech-to-text-benchmark)
- [TTS Arena — Hugging Face](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2)
- [Blizzard Challenge](https://www.synsig.org/index.php/Blizzard_Challenge)
- [VoiceMOS Challenge 2023](https://voicemos-challenge-2023.github.io/)
- [UTMOS paper (arXiv:2204.02152)](https://arxiv.org/abs/2204.02152)
- [NaturalSpeech 3 (arXiv:2403.03100)](https://arxiv.org/abs/2403.03100)
- [VALL-E 2 (arXiv:2406.05370)](https://arxiv.org/abs/2406.05370)
- [Distil-Whisper large-v3 model card](https://huggingface.co/distil-whisper/distil-large-v3)
- [NVIDIA Parakeet-TDT-0.6B-v2 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA Canary-Qwen-2.5B model card](https://huggingface.co/nvidia/canary-qwen-2.5b)
- [IBM Granite-Speech-3.3-8B model card](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- [WebMUSHRA](https://github.com/audiolabs/webMUSHRA)
- [VERSA toolkit](https://github.com/shinjiwlab/versa)
- [jiwer library](https://github.com/jitsi/jiwer)
- [Papers With Code — Speech Synthesis](https://paperswithcode.com/task/speech-synthesis)
- [SUPERB Benchmark](https://superbbenchmark.org/)
- [ESB datasets](https://huggingface.co/datasets/esb/datasets)

