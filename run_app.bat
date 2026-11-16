@echo off
REM Windows launcher. Edit the path below, then double-click.
set XBCI_ARTIFACT_ROOT=C:\path\to\XBCI_BNCI2014_001
if not exist .venv ( python -m venv .venv && call .venv\Scripts\activate && pip install -r requirements.txt ) else ( call .venv\Scripts\activate )
streamlit run app.py
