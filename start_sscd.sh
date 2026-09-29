#!/usr/bin/env sh
# Start the SSCD web app (Linux / macOS):  ./start_sscd.sh
# The first start installs the Python environment (a few minutes).
# Extra arguments are passed on to Streamlit, e.g. ./start_sscd.sh --server.port 8502
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    echo "e.g.:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Preparing the SSCD environment (the first time this installs the packages - a few minutes)..."
uv sync --extra gui

echo "Starting SSCD - open the address shown below in your browser. Press Ctrl+C to stop the app"
echo "(a run that is in progress keeps going)."
exec uv run --no-sync python -m streamlit run sscd_app.py "$@"
