#!/bin/bash
# run.sh - One-click launcher for Grape Analyzer on Mac
# Handles: Java install check, venv, pip install, app launch

echo ""
echo "🍇 ============================================"
echo "   Grape Analyzer - Starting Setup"
echo "============================================"
echo ""

# ── Check Java ──
echo "☕ Checking Java..."
if ! java -version &>/dev/null; then
    echo "⚠️  Java not found. Installing via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "❌ Homebrew not found. Install it first:"
        echo '   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    brew install openjdk
    export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
    export JAVA_HOME="/opt/homebrew/opt/openjdk"
    echo "✅ Java installed"
else
    echo "✅ Java found"
fi

# Set JAVA_HOME if not set
if [ -z "$JAVA_HOME" ]; then
    if [ -d "/opt/homebrew/opt/openjdk" ]; then
        export JAVA_HOME="/opt/homebrew/opt/openjdk"
        export PATH="$JAVA_HOME/bin:$PATH"
    elif [ -d "/usr/local/opt/openjdk" ]; then
        export JAVA_HOME="/usr/local/opt/openjdk"
        export PATH="$JAVA_HOME/bin:$PATH"
    fi
fi

# ── Go to project folder ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Virtual environment ──
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "📦 Activating virtual environment..."
source .venv/bin/activate

# ── Install dependencies ──
echo "📦 Installing/updating dependencies (first time may take a few minutes)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete"
echo "ℹ️  First run: Fiji will download automatically (~300MB, one time only)"
echo ""

# ── Launch app ──
echo "🚀 Launching Grape Analyzer GUI..."
echo ""
python -m app.main
