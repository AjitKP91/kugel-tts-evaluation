# Evaluating `kugel-3` (KugelAudio TTS) in this harness

End-to-end guide for running the TTS test suite (2.1, 2.2, 2.3, 2.5, 2.6, 2.7,
2.8 — signal quality 2.4 is retired, see §4a) against KugelAudio's `kugel-3`
model. The harness reuses the same test modules and local scoring tools
(Whisper, UTMOS/DNSMOS, ECAPA-TDNN) as the original multi-provider evaluation;
only the TTS client changes.

Source: [KugelAudio docs](https://docs.kugelaudio.com/).

---

## 1. What KugelAudio provides

KugelAudio is **TTS-only** — there is no speech-to-text/ASR product, so this
repo has no STT suite. It exposes two transports (no gRPC):

- **Batch REST** — `POST https://api.kugelaudio.com/v1/tts/generate`. Returns
  raw PCM16 LE audio (headers `X-Sample-Rate`, `X-Audio-Format: pcm_s16le`).
- **WebSocket streaming** — `wss://api.kugelaudio.com/ws/tts?api_key=...`. Sends
  one JSON request, streams base64 PCM16 chunks, ends with a `final` message.

Native sample rate is 24 kHz (8000/16000/22050/24000/44100 also accepted).
Models: `kugel-3` (default), `kugel-2.5`, `kugel-2-turbo`. 39 languages. Input
limit 10,000 characters.

---

## 2. Get credentials

1. Create an API key in your KugelAudio dashboard.
2. `bash scripts/setup.sh` prompts for it (input hidden) and saves it to a
   gitignored `.env` — the key is read from there at call time and never
   committed. You don't export it manually. To change it later, delete the
   `KUGELAUDIO_API_KEY=` line in `.env` and re-run setup, or edit `.env`.
3. Pick the `voice_id` to evaluate from the Voices section of the docs.

Quick manual smoke check (load the saved key into your shell first):

```bash
set -a; source .env; set +a
curl -X POST "https://api.kugelaudio.com/v1/tts/generate" \
  -H "Authorization: Bearer $KUGELAUDIO_API_KEY" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{"text":"Hello from Kugel.","model_id":"kugel-3","voice_id":1071}' \
  --output smoke.pcm
```

---

## 3. Configure

`eval/config.yaml` (already set to `provider: kugel`):

```yaml
tts:
  provider: kugel
  kugel:
    api_key_env: KUGELAUDIO_API_KEY
    model_id: kugel-3          # or kugel-2.5 / kugel-2-turbo
    voice_id: 1071
    language: null             # ISO 639-1 (e.g. "en"); null = auto-detect
    cfg_scale: 2.0             # 1.2–2.5, higher = more expressive
    temperature: 0.4
    speed: 1.0                 # 0.8–1.2
    normalize: true
    sample_rate: 24000
    rest_endpoint: https://api.kugelaudio.com/v1/tts/generate
    ws_endpoint: wss://api.kugelaudio.com/ws/tts
    voices_endpoint: https://api.kugelaudio.com/v1/voices
    request_timeout_s: 120
    reference_set_dir: null    # dormant — used only by clone_reference_voice.py
    reference_voice_id: null   # dormant — see §4a
```

---

## 4a. Test 2.4 (Signal Quality) — retired

MCD/PESQ/STOI only make sense when the synthesized audio and the reference
recording are the **same speaker, saying the same words, aligned in time**. A
synthetic TTS voice has no ground-truth human recording, so there is nothing
valid to compare against:

- We prototyped a workaround: clone a single-speaker corpus (LJSpeech) into a
  Kugel voice and compare against the real LJSpeech recordings. Even so, MCD
  stayed unreliable (wrong cepstral representation for the metric) and PESQ/STOI
  collapsed to their floor (they need sample-level alignment, impossible for TTS
  vs a human take).
- **Test 2.4 is therefore not part of the suite.** Audio quality is covered by
  **2.1** (naturalness), **2.2** (intelligibility), and **2.8** (voice
  consistency).

The clone tooling (`scripts/clone_reference_voice.py` and `clone_voice()` in the
client) stays in the repo but **dormant** — not run by setup, not wired into any
test. It's kept for future use if Kugel provides ground-truth reference
recordings per voice, which would make signal-quality metrics valid. To use it
manually: `set -a; source .env; set +a` then
`python scripts/clone_reference_voice.py`.

---

## 4. How each test maps to KugelAudio

| Test | Interface used | Kugel notes |
|------|----------------|-------------|
| 2.1 Naturalness | REST batch | Unchanged — UTMOS/DNSMOS scored locally |
| 2.2 Intelligibility | REST batch | Round-trip WER via local Whisper |
| 2.3 Prosody | REST batch | Reference-free F0 + WPM |
| 2.5 Streaming TTFB & RTF | **WebSocket** vs REST | WebSocket gives true TTFB; REST is chunked read |
| 2.6 Throughput & Concurrency | REST (thread-pool + async) | `tts_concurrency_levels` = [1,5,10,20] |
| 2.7 Edge Cases | REST batch | SSML/markup cases are plain-text robustness probes (Kugel has no documented SSML) |
| 2.8 Long-Form Consistency | REST batch | ECAPA-TDNN speaker-similarity drift |

The client (`eval/tts/kugel_client.py`) mirrors the original `TTSClient` public
surface — `synthesize_batch`, `synthesize_batch_rest`, `synthesize_stream`
(WebSocket), `synthesize_stream_rest`, `save_synthesis`, `bytes_to_wav` — plus
`clone_voice` (dormant; used only by the voice-clone tooling) — so the test
modules call it through `build_tts_client(config)` without branching.

---

## 5. Run

```bash
# Verify connectivity + schema first
python -m eval.run phase0

# Single tests while validating
python -m eval.run tts --test naturalness
python -m eval.run tts --test latency        # exercises WebSocket + REST

# Full run + report
python -m eval.run all
```

On the VM, prefer `bash scripts/start_eval.sh` (tmux-persistent, prompts for the
API key, writes to `results/run-<DD-MM-YY>/kugel/`).

---

## 6. Known caveats

- **Signal quality (former Test 2.4)** is retired — see §4a. MCD/PESQ/STOI need
  a same-speaker parallel reference that a TTS service can't provide. Quality is
  covered by 2.1 (naturalness), 2.2 (intelligibility), and 2.8 (consistency).
- **SSML** is not documented for KugelAudio; edge-case SSML inputs are scored as
  plain-text robustness (a pass means the tags didn't error, not that they were
  interpreted).
- **Sample rate**: audio is generated natively at 24 kHz; other rates use
  server-side resampling. Keep `sample_rate: 24000` unless you specifically want
  to test resampling behaviour.
