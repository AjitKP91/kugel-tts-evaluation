#!/usr/bin/env bash
# Start the evaluation in a persistent tmux session.
# Writes into a dated per-provider dir: results/run-<DD-MM-YY>/kugel/
# so historical runs stay untouched.
# Asks for the KugelAudio API key, then launches inside tmux so SSH drops don't kill it.
# Usage: bash scripts/start_eval.sh [all|tts|phase0|report]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
COMMAND="${1:-all}"
SESSION="eval"

echo "=========================================="
echo "  Kugel TTS Eval — Start Run ($COMMAND)"
echo "=========================================="

# ── Check venv exists ────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "ERROR: .venv not found. Run setup first: bash scripts/setup.sh"
    exit 1
fi

# ── Pull latest code ─────────────────────────────────────────────────────────
echo ""
echo "Pulling latest code..."
git pull

# ── Ask for KugelAudio API key ───────────────────────────────────────────────
echo ""
read -rsp "Paste your KUGELAUDIO_API_KEY (input hidden): " KUGEL_KEY
echo ""
if [ -z "$KUGEL_KEY" ]; then
    echo "ERROR: API key cannot be empty."
    exit 1
fi

# ── Compute per-run, per-provider results directory ──────────────────────────
# Landing at results/run-<DD-MM-YY>/kugel/ keeps each date's run isolated. If a
# run spans midnight, RUN_DATE is set once at launch and doesn't roll over —
# resume writes to the same dir.
RUN_DATE="$(date +%d-%m-%y)"
RESULTS_DIR="results/run-${RUN_DATE}/kugel"
mkdir -p "$RESULTS_DIR"
echo ""
echo "Results dir: $RESULTS_DIR"

# ── Kill existing tmux session if running ────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Killing existing tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION"
fi

# ── Write a launcher script for inside tmux ──────────────────────────────────
LAUNCHER="$REPO_DIR/scripts/_tmux_launcher.sh"
cat > "$LAUNCHER" <<LAUNCHER_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
source "$REPO_DIR/.venv/bin/activate"
export HF_HOME=~/hf_home
export TORCH_HOME=~/torch_home
export PIP_CACHE_DIR=~/pip_cache
export KUGELAUDIO_API_KEY="$KUGEL_KEY"
# Make PyTorch bundled CUDA visible (fixes UTMOS libcudart error)
TORCH_LIB="\$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="\$TORCH_LIB:\${LD_LIBRARY_PATH:-}"
mkdir -p "$RESULTS_DIR"
echo "Starting: python -m eval.run $COMMAND --results-dir $RESULTS_DIR"
echo "Log: $RESULTS_DIR/run.log (append mode)"
python -m eval.run $COMMAND --results-dir "$RESULTS_DIR" 2>&1 | tee -a "$RESULTS_DIR/run.log"
echo ""
echo "=== Run complete. Press any key to close. ==="
read -n1
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# ── Start tmux session ───────────────────────────────────────────────────────
echo ""
echo "Starting tmux session '$SESSION'..."
tmux new-session -d -s "$SESSION" "bash $LAUNCHER"

echo ""
echo "=========================================="
echo "  Eval is running in tmux session '$SESSION'."
echo ""
echo "  Attach to watch progress:"
echo "    tmux attach -t $SESSION"
echo ""
echo "  Detach without killing:"
echo "    Ctrl+B then D"
echo ""
echo "  Tail the log from outside tmux:"
echo "    tail -f $REPO_DIR/$RESULTS_DIR/run.log"
echo "=========================================="
