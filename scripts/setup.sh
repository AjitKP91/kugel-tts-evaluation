#!/usr/bin/env bash
# One-time setup: system packages, /mnt caches, Python venv, dependencies, HuggingFace login.
# Run this once after cloning the repo.
# Usage: bash scripts/setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=========================================="
echo "  Kugel TTS Eval — One-Time Setup"
echo "=========================================="

# ── 1. System packages ──────────────────────────────────────────────────────
echo ""
echo "[1/8] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg libavcodec-extra libsndfile1 git python3 python3-pip python3-venv

# libopenh264: ffmpeg needs this for H.264; version on the system may differ
sudo apt-get install -y libopenh264-dev 2>/dev/null || true
# If only .so.6 is available, symlink it as .so.5 (what ffmpeg looks for)
if [ ! -f /usr/lib/x86_64-linux-gnu/libopenh264.so.5 ]; then
    SO6=$(find /usr/lib -name "libopenh264.so.6" 2>/dev/null | head -1)
    if [ -n "$SO6" ]; then
        sudo ln -sf "$SO6" /usr/lib/x86_64-linux-gnu/libopenh264.so.5
        echo "  Symlinked libopenh264.so.6 → libopenh264.so.5"
    fi
fi

# ── 2. Cache directories in home (OS disk now 512GB) ────────────────────────
echo ""
echo "[2/8] Setting up cache directories..."
mkdir -p ~/hf_home ~/torch_home ~/pip_cache

# Persist env vars in ~/.bashrc (idempotent)
for line in \
    'export HF_HOME=~/hf_home' \
    'export TORCH_HOME=~/torch_home' \
    'export PIP_CACHE_DIR=~/pip_cache'; do
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done
export HF_HOME=~/hf_home
export TORCH_HOME=~/torch_home
export PIP_CACHE_DIR=~/pip_cache
echo "  Cache dirs: HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME"

# ── 3. Python venv ──────────────────────────────────────────────────────────
echo ""
echo "[3/8] Creating Python virtual environment..."
if [ ! -d "$REPO_DIR/.venv" ]; then
    python3 -m venv "$REPO_DIR/.venv"
fi
source "$REPO_DIR/.venv/bin/activate"

# ── 4. Install Python dependencies ──────────────────────────────────────────
echo ""
echo "[4/8] Installing Python dependencies (this may take 10-20 minutes)..."

# Install uv — much faster resolver, handles complex graphs pip cannot
pip install uv --no-cache-dir -q
echo "  uv installed."

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.x drivers).
# Pinned to 2.2.1 to match nisqa's strict torch==2.2.1 requirement.
# Must be done BEFORE requirements.txt so other packages (utmos, whisper,
# speechbrain) pick up the GPU-enabled torch instead of the CPU-only PyPI wheel.
echo "  Installing PyTorch 2.2.1 (CUDA 12.1)..."
uv pip install "torch==2.2.1" "torchvision==0.17.1" "torchaudio==2.2.1" \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir
python -c "import torch; print('  torch CUDA available:', torch.cuda.is_available())"

# Pre-fix omegaconf: fairseq/utmos ships 2.0.6 which pip 24+ rejects
echo "  Pre-fixing omegaconf..."
uv pip install "omegaconf>=2.1" --no-cache-dir

# Install all requirements via uv (avoids resolution-too-deep errors)
uv pip install -r "$REPO_DIR/requirements.txt" --no-cache-dir

# speechmetrics: not on PyPI and declares numpy<1.24 which conflicts with nisqa.
# Install with --no-deps to bypass the stale constraint — works fine with numpy 1.26.4.
echo "  Installing speechmetrics (--no-deps to skip stale numpy<1.24 constraint)..."
pip install --no-deps git+https://github.com/aliutkus/speechmetrics --no-cache-dir

# Ensure setuptools is present for pyworld
uv pip install setuptools --no-cache-dir

echo "  Dependencies installed."

# ── Set LD_LIBRARY_PATH for PyTorch bundled CUDA (fixes UTMOS libcudart error) ──
TORCH_LIB="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null)/lib"
if [ -d "$TORCH_LIB" ]; then
    LINE="export LD_LIBRARY_PATH=\"$TORCH_LIB:\$LD_LIBRARY_PATH\""
    grep -qxF "$LINE" ~/.bashrc || echo "$LINE" >> ~/.bashrc
    export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
    echo "  CUDA runtime path set: $TORCH_LIB"
fi

# ── 5. HuggingFace login ─────────────────────────────────────────────────────
echo ""
echo "[5/8] HuggingFace login (optional)"
echo "  Only needed for the optional LJSpeech reference set used by Test 2.4,"
echo "  which self-skips for Kugel unless a matched-speaker reference is set."
echo "  Get your token at: https://huggingface.co/settings/tokens"
echo ""
hf auth login || echo "  Skipped HuggingFace login (fine — LJSpeech is optional)."
echo "  HuggingFace step done."

# ── 6. KugelAudio API key ────────────────────────────────────────────────────
# Prompt once and persist to a gitignored .env so you never have to export it
# by hand. start_eval.sh and manual runs read this file automatically.
echo ""
echo "[6/8] KugelAudio API key"
ENV_FILE="$REPO_DIR/.env"
if [ -f "$ENV_FILE" ] && grep -q '^KUGELAUDIO_API_KEY=' "$ENV_FILE"; then
    echo "  .env already contains KUGELAUDIO_API_KEY — keeping it."
    echo "  (Delete the line in $ENV_FILE and re-run setup to change it.)"
else
    read -rsp "  Paste your KUGELAUDIO_API_KEY (input hidden): " KUGEL_KEY
    echo ""
    if [ -z "$KUGEL_KEY" ]; then
        echo "  WARNING: no key entered. You can add it later with:"
        echo "    echo 'KUGELAUDIO_API_KEY=<your-key>' >> $ENV_FILE"
    else
        touch "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        # Remove any stale line, then append the new one.
        grep -v '^KUGELAUDIO_API_KEY=' "$ENV_FILE" > "$ENV_FILE.tmp" 2>/dev/null || true
        mv "$ENV_FILE.tmp" "$ENV_FILE" 2>/dev/null || true
        echo "KUGELAUDIO_API_KEY=$KUGEL_KEY" >> "$ENV_FILE"
        echo "  Saved to $ENV_FILE (gitignored, chmod 600)."
    fi
fi

# ── 7. Test 2.4 reference (clone a matched-speaker voice) ────────────────────
# Test 2.4 (MCD/PESQ/STOI) needs a same-speaker reference. Clone the LJSpeech
# speaker into a Kugel voice now so 2.4 is enabled by default. Non-fatal: if the
# API key or HF access is missing, Test 2.4 simply self-skips at run time.
echo ""
echo "[7/8] Test 2.4 matched-speaker reference (voice clone)"
if [ -f "$ENV_FILE" ] && grep -q '^KUGELAUDIO_API_KEY=' "$ENV_FILE"; then
    set -a; source "$ENV_FILE"; set +a
    echo "  Cloning LJSpeech into a Kugel voice and wiring it into config.yaml..."
    if python scripts/clone_reference_voice.py; then
        echo "  Test 2.4 reference ready."
    else
        echo "  WARNING: clone step failed — Test 2.4 will self-skip until you run"
        echo "           'python scripts/clone_reference_voice.py' manually."
    fi
else
    echo "  Skipped — no API key in .env. Test 2.4 will self-skip. Enable later with:"
    echo "    python scripts/clone_reference_voice.py"
fi

# ── 8. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "[8/8] Verifying setup..."
python -c "import websocket; print('  websocket-client OK')"
python -c "import jiwer; print('  jiwer OK')"
python -c "import soundfile; print('  soundfile OK')"
python -c "import whisper; print('  openai-whisper OK')"
python -c "import torch; avail = torch.cuda.is_available(); print('  torch CUDA:', avail, '(' + (torch.version.cuda or 'CPU only') + ')')"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Your API key is saved in .env — no need to export it."
echo "  Next step, run the evaluation:"
echo "    bash scripts/start_eval.sh"
echo "=========================================="
