#!/usr/bin/env bash
# Download the optional LJSpeech reference set and cache it in ~/hf_home.
# Only needed for Test 2.4 (signal quality), which self-skips for Kugel unless
# a matched-speaker reference set is configured. All other TTS test text is
# bundled in eval/data/ — no download required.
# Usage: bash scripts/download_datasets.sh [ljspeech|all]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASET="${1:-all}"

echo "=========================================="
echo "  Kugel TTS Eval — Download Datasets"
echo "  Dataset: $DATASET"
echo "=========================================="

# ── Check venv exists ────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "ERROR: .venv not found. Run setup first: bash scripts/setup.sh"
    exit 1
fi

source "$REPO_DIR/.venv/bin/activate"
export HF_HOME=~/hf_home
export TORCH_HOME=~/torch_home
export PIP_CACHE_DIR=~/pip_cache

echo ""
echo "HF_HOME: $HF_HOME"
echo ""

# ── Download ─────────────────────────────────────────────────────────────────
python -m eval.data.download_datasets --dataset "$DATASET"

echo ""
echo "=========================================="
echo "  Download complete."
echo "  Datasets cached in: $HF_HOME/datasets"
echo "  Run 'bash scripts/start_eval.sh' to start evaluation."
echo "=========================================="
