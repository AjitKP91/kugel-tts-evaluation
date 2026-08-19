# Evaluation Runbook — Kugel-TTS

## TL;DR

**Run on the Azure VM, not your laptop.**
Running locally adds network jitter to every API call, which contaminates all
latency and RTF measurements. The VM also has the GPU needed for the local
scoring tools (Whisper, UTMOS, ECAPA-TDNN).

---

## 1. Prerequisites

### 1.1 KugelAudio API key
1. Get an API key from your KugelAudio dashboard (`kugelaudio.com`).
2. Pick the `voice_id` you want to evaluate (see the Voices section of
   `docs.kugelaudio.com`; the config default is `1071`).

### 1.2 HuggingFace (optional)
Only Test 2.4 (signal quality) uses an external dataset (LJSpeech), and it
self-skips for Kugel unless you configure a matched-speaker reference set. If
you want it, create a free account at https://huggingface.co and generate a
read token. Everything else (Harvard sentences, intelligibility / edge-case /
long-form text) is bundled in `eval/data/`.

---

## 2. Where to Run What

| Task | Where | Why |
|------|-------|-----|
| All eval tests (phase0, tts) | **Azure VM** | Clean latency numbers, GPU for scoring |
| Whisper transcription (local GPU) | **Azure VM** | Handles large-v3 in real time |
| UTMOS / SpeechBrain scoring | **Azure VM** | GPU-accelerated; slow on CPU |
| Viewing the HTML report | Laptop | Copy `results/report.html` back via SCP |
| Editing config / fixing code | Either | Up to you |

---

## 3. One-Time VM Setup

### 3.1 Get the repo onto the VM

```bash
git clone <your-repo-url> kugel-tts-evaluation
cd kugel-tts-evaluation
```

### 3.2 Run setup

```bash
bash scripts/setup.sh
```

This installs system packages, creates `.venv`, installs all Python
dependencies via `uv`, and sets `LD_LIBRARY_PATH` for PyTorch's bundled CUDA
(needed by UTMOS). Takes 10–20 minutes on first run.

### 3.3 (Optional) pre-download the LJSpeech reference

Only needed if you plan to enable Test 2.4 with a matched-speaker reference:

```bash
bash scripts/download_datasets.sh ljspeech
```

---

## 4. Configure

Edit `eval/config.yaml`:

```yaml
tts:
  provider: kugel
  kugel:
    api_key_env: KUGELAUDIO_API_KEY
    model_id: kugel-3
    voice_id: 1071
    sample_rate: 24000
    rest_endpoint: https://api.kugelaudio.com/v1/tts/generate
    ws_endpoint: wss://api.kugelaudio.com/ws/tts
    request_timeout_s: 120
```

Then export your key:

```bash
export KUGELAUDIO_API_KEY="..."
```

---

## 5. Running the Evaluation

`start_eval.sh` pulls latest code, prompts for your API key, and runs inside a
persistent tmux session so SSH disconnects don't interrupt it.

```bash
bash scripts/start_eval.sh           # everything (phase0 + tts + report)
bash scripts/start_eval.sh tts       # TTS tests only
bash scripts/start_eval.sh phase0    # connectivity check only
bash scripts/start_eval.sh report    # regenerate report from existing results
```

Monitor and re-attach:

```bash
tail -f results/run-<DD-MM-YY>/kugel/run.log
tmux attach -t eval          # detach with Ctrl+B then D
```

### Run manually (single test)

```bash
source .venv/bin/activate
export KUGELAUDIO_API_KEY="..."

python -m eval.run tts --test naturalness
python -m eval.run tts --test latency
python -m eval.run all --dry-run       # verify config, no API calls
```

Available TTS test names:
`naturalness`, `intelligibility`, `prosody`, `signal_quality`, `latency`,
`concurrency`, `edge_cases`, `long_form`

---

## 6. Resuming After Interruption

Every API-call result is written to a `.jsonl` file immediately. Re-running the
same command **skips already-completed items** — no data lost, no duplicate
calls.

---

## 7. Outputs

```
results/run-<DD-MM-YY>/kugel/
├── phase0/discovery.json
├── tts/
│   ├── naturalness/{calls.jsonl, summary.csv}
│   ├── latency/{calls.jsonl, summary.csv}
│   └── ...                       # one folder per test
└── report.html
```

Copy the report to your laptop:

```bash
scp <vm-user>@<vm-ip>:~/kugel-tts-evaluation/results/run-*/kugel/report.html ~/Desktop/
```

---

## 8. Cleaning Up

```bash
bash scripts/clean.sh          # clear results, keep .venv + datasets
bash scripts/clean.sh --full   # also remove .venv (re-run setup.sh after)
```

---

## 9. Troubleshooting

### `Set $KUGELAUDIO_API_KEY with a valid KugelAudio API key`
```bash
export KUGELAUDIO_API_KEY="..."
```

### `websocket-client` import error (Test 2.5 streaming)
```bash
.venv/bin/pip install websocket-client --no-cache-dir
```

### 401 / 403 from the API
Check the API key is valid and the `voice_id` / `model_id` in `config.yaml`
exist. A 404 usually means an invalid `voice_id`.

### UTMOS `libcudart.so.*: cannot open shared object file`
PyTorch ships its own CUDA runtime. Add it to `LD_LIBRARY_PATH`:
```bash
TORCH_LIB=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
```
`setup.sh` and `start_eval.sh` set this automatically.

### `ModuleNotFoundError: No module named 'pkg_resources'` (pyworld)
```bash
.venv/bin/pip install setuptools --no-cache-dir
```
F0 metrics in Test 2.3 are skipped automatically if pyworld is unavailable.

### `ffmpeg not found` (pydub / librosa)
```bash
sudo apt-get install -y ffmpeg
```

### Out of disk space in results/
Synthesized `.wav` files are deleted after scoring, but if a run is interrupted
some may remain:
```bash
find results -name '*.wav' -delete
```
