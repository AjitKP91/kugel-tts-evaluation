#!/usr/bin/env bash
# Clean the Kugel TTS Eval harness — removes all data, results, and model caches.
# .venv is kept so setup.sh does not need to be re-run.
#
# Usage: bash scripts/clean.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=========================================="
echo "  Kugel TTS Eval — Clean"
echo "=========================================="

echo ""
echo "Disk usage BEFORE clean:"
df -h /
echo ""
echo "Top space users:"
du -sh "$HOME"/hf_home "$HOME"/torch_home "$HOME"/pip_cache \
        "$REPO_DIR/.venv" "$REPO_DIR/results" 2>/dev/null \
    | sort -rh | head -10 || true
echo ""

# Kill tmux eval sessions
for session in eval eval_stt eval_tts; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Killing tmux session '$session'..."
        tmux kill-session -t "$session"
    fi
done

# Stray temp files from crashed runs
echo "Removing stray /tmp/eval_ds_* and /tmp/*.wav..."
rm -rf /tmp/eval_ds_* 2>/dev/null || true
rm -f /tmp/*.wav 2>/dev/null || true

# Results
echo "Removing results/..."
rm -rf "$REPO_DIR/results"
mkdir -p "$REPO_DIR/results"
rm -f "$REPO_DIR/scripts/_tmux_launcher.sh"

# HuggingFace cache (datasets and model weights — token file is preserved)
echo "Clearing HuggingFace dataset and model cache (~/hf_home/datasets, ~/hf_home/hub)..."
rm -rf "$HOME/hf_home/datasets"
rm -rf "$HOME/hf_home/hub"
rm -rf "$HOME/hf_home/modules"

# PyTorch hub cache
echo "Clearing PyTorch hub cache (~/torch_home)..."
rm -rf "$HOME/torch_home"

# Whisper model cache (deleted in-program too, but clean up any leftovers)
echo "Clearing Whisper model cache (~/.cache/whisper)..."
rm -rf "$HOME/.cache/whisper"

# Pip wheel cache
echo "Clearing pip wheel cache..."
rm -rf "$HOME/pip_cache"/* 2>/dev/null || true

echo ""
echo "Disk usage AFTER clean:"
df -h /
echo ""
echo "Top space users:"
du -sh "$HOME"/hf_home "$HOME"/torch_home "$HOME"/pip_cache \
        "$REPO_DIR/.venv" "$REPO_DIR/results" 2>/dev/null \
    | sort -rh | head -10 || true

echo ""
echo "=========================================="
echo "  Clean done."
echo "  .venv kept — run 'bash scripts/start_eval.sh' to restart."
echo "=========================================="
