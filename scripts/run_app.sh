#!/usr/bin/env bash
# Convenience launcher: prepare demo data + index, then start the Streamlit app.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Generating demo data (if needed)..."
python scripts/make_demo_data.py

echo "==> Building retrieval index (fallback backend)..."
python scripts/build_index.py --dataset demo --embedding-backend fallback

echo "==> Launching Streamlit app..."
echo "    Research demo only. Not for diagnosis or clinical use."
streamlit run app/streamlit_app.py
