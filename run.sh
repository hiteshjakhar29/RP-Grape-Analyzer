#!/bin/bash
# run.sh - One-click launcher for Grape Analyzer on Mac
# Handles: venv setup, pip install, app launch

echo ""
echo "============================================"
echo "   Grape Analyzer - Starting Setup"
echo "============================================"
echo ""

# ── Go to project folder ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Virtual environment ──
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# ── Install dependencies ──
echo "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Setup complete."
echo "Note: Fiji must be installed at /Applications/FijiWorking/Fiji.app"
echo "      or locate it via the Settings dialog on first run."
echo ""

# ── Launch app ──
echo "Launching Grape Analyzer..."
echo ""
python main.py
