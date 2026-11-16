#!/usr/bin/env bash
# macOS / Linux launcher.  Usage:  XBCI_ARTIFACT_ROOT=/path/to/XBCI_BNCI2014_001 ./run_app.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt; else . .venv/bin/activate; fi
streamlit run app.py
