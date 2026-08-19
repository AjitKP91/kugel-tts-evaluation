# Evaluation Runbook — Kugel-TTS

## TL;DR

**Run on the Azure VM, not your laptop.**
Running locally adds network jitter to every API call, which contaminates all
latency and RTF measurements. The VM also has the GPU needed for the local
scoring tools (Whisper, UTMOS, ECAPA-TDNN).

---

## 1. Prerequisites

### 1.1 KugelAudio API key
Get an API key from your KugelAudio dashboard (`kugelaudio.com`). **You do not
need to export it** — `scripts/setup.sh` prompts for it once and saves it to a
gitignored `.env` file; every run reads it from there automatically (see §3.2).

Also pick the `voice_id` you want to evaluate (Voices section of
`docs.kugelaudio.com`; the config default is `1071`).

### 1.2 HuggingFace (only for Test 2.4)
Test 2.4 (signal quality) is the only test that uses an external dataset
(LJSpeech), and only when you enable it (§3.4). If you plan to run it, create a
free account at https://huggingface.co and generate a read token — `setup.sh`
runs `hf auth login` for you. Everything else (Harvard sentences,
intelligibility / edge-case / long-form text) is bundled in `eval/data/`.

---

## 2. Where to Run What

| Task | Where | Why |
|------|-------|-----|
| All eval tests (phase0, tts) | **Azure VM** | Clean latency numbers, GPU for scoring |
| Whisper transcription (local GPU) | **Azure VM** | Handles large-v3 in real time |
| UTMOS / SpeechBrain scoring | **Azure VM** | GPU-accelerated; slow on CPU |
| Viewing the HTML report | Laptop | Copy `report.html` back via SCP |
| Editing config / fixing code | Either | Up to you |

---

## 3. One-Time VM Setup

### 3.1 Get the repo onto the VM

```bash
git clone <your-repo-url> kugel-tts-evaluation
cd kugel-tts-evaluation
```

### 3.2 Run setup (this is where you enter the API key)

```bash
bash scripts/setup.sh
```

The script, in order:
1. Installs system packages (`ffmpeg`, `libsndfile1`, …).
2. Creates `.venv` and installs all Python dependencies via `uv`.
3. Sets `LD_LIBRARY_PATH` to PyTorch's bundled CUDA (needed by UTMOS).
4. Runs `hf auth login` (optional — skip if not using Test 2.4).
5. **Prompts for your `KUGELAUDIO_API_KEY` (input hidden) and saves it to
   `.env`** with `chmod 600`. `.env` is gitignored, so the key never leaves the
   VM. If `.env` already has a key it is kept — delete that line and re-run
   setup to change it.
6. Verifies key imports.

Takes 10–20 minutes on first run. **After this you never type or export the key
again** — `start_eval.sh` and manual runs load it from `.env`.

### 3.3 Set your voice / model in `eval/config.yaml`

```yaml
tts:
  provider: kugel
  kugel:
    api_key_env: KUGELAUDIO_API_KEY   # name of the env var; value lives in .env
    model_id: kugel-3
    voice_id: 1071
    sample_rate: 24000
    rest_endpoint: https://api.kugelaudio.com/v1/tts/generate
    ws_endpoint: wss://api.kugelaudio.com/ws/tts
    request_timeout_s: 120
```

You normally only change `voice_id` and `model_id`. Do **not** put the key in
this file — `api_key_env` just names the variable; the value is read from `.env`
(or the environment) at call time.

### 3.4 (Optional) Enable Test 2.4 — matched-speaker reference

Test 2.4 (MCD/PESQ/STOI) self-skips unless you build a same-speaker reference.
One command clones a single-speaker corpus (LJSpeech) into a Kugel voice and
saves the reference recordings:

```bash
source .venv/bin/activate
set -a; source .env; set +a          # load the API key into this shell
python scripts/clone_reference_voice.py --n-clone 6 --n-reference 30 \
    --out eval/data/kugel_reference
```

It prints two values — copy them into `eval/config.yaml` under `tts.kugel`:

```yaml
    reference_set_dir: eval/data/kugel_reference
    reference_voice_id: <printed voice_id>
```

> This measures **clone fidelity** (how faithfully Kugel reproduces the cloned
> speaker), not stock-voice quality. Skip this step to leave Test 2.4 skipping.

---

## 4. Running the Evaluation

`start_eval.sh` pulls latest code, **loads the API key from `.env`** (only
prompts if it's missing), and runs inside a persistent tmux session so SSH
disconnects don't interrupt it.

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

The key is in `.env`; load it into your shell first, then run:

```bash
source .venv/bin/activate
set -a; source .env; set +a          # exports KUGELAUDIO_API_KEY for this shell

python -m eval.run phase0               # connectivity + schema
python -m eval.run tts --test naturalness
python -m eval.run tts --test latency   # exercises WebSocket + REST
python -m eval.run all --dry-run        # verify config, no API calls
```

Available TTS test names:
`naturalness`, `intelligibility`, `prosody`, `signal_quality`, `latency`,
`concurrency`, `edge_cases`, `long_form`

---

## 5. Resuming After Interruption

Every API-call result is written to a `.jsonl` file immediately. Re-running the
same command **skips already-completed items** — no data lost, no duplicate
calls.

---

## 6. Outputs

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

## 7. Cleaning Up

```bash
bash scripts/clean.sh          # clear results, keep .venv + datasets
bash scripts/clean.sh --full   # also remove .venv (re-run setup.sh after)
```

---

## 8. Troubleshooting

### `Set $KUGELAUDIO_API_KEY with a valid KugelAudio API key`
The key isn't in your environment. Either re-run `bash scripts/setup.sh` (which
saves it to `.env`), or load an existing `.env` into your shell:
```bash
set -a; source .env; set +a
```
`start_eval.sh` does this automatically — this only bites manual runs.

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
