> This file mirrors `.github/copilot-instructions.md` and `CLAUDE.md`; keep all three in sync when any changes.

# AGENTS.md — pltr-equity-research-project

## Purpose
Automated PLTR equity research pipeline with strategy backtesting, risk analytics, factor modeling, Monte Carlo/GARCH forecasting, and PDF report generation.

## Tech stack
- Python; dependencies in `requirements.txt`
- Key libraries: `pandas`, `numpy`, `matplotlib`, `yfinance`, `scipy`, `statsmodels`, `arch`, `requests`

## Folder structure
- `config/` (1 files)
- `data/` (3 files)
- `outputs/` (7 files)
- `src/` (20 files)
- `README.md`
- `main.py`
- `requirements.txt`

## Setup, run, test
Commands documented in README.md (referenced files exist):
```bash
pip install -r requirements.txt
python main.py
```
- No automated tests exist yet; if adding them, use pytest under `tests/`.

## Conventions observed
- Config lives in: `config/config.py`; avoid hardcoding parameters elsewhere.
- Random seeds are set for reproducibility (Monte Carlo/simulation); keep seeded, deterministic runs.
- Raw data (`data/raw/`) is gitignored; download/fetch it via the pipeline, do not commit it.
- Public data sources referenced: yfinance.
- Write paths relative to the repo root (no absolute local paths).

## Output artifacts
- Generated outputs go to: `outputs/charts/`, `outputs/tables/`
- A PDF report is produced with `reportlab`; keep it in sync with model changes.
- Charts are generated with matplotlib.

## Guardrails
- Never commit secrets, API keys, tokens, or `.env` files.
- Do not commit large raw datasets or binaries; keep data fetched or documented with its source.
- Keep analysis source-backed: cite data sources and assumptions; label scenario/forecast outputs as model estimates, not facts.
- Preserve reproducibility: no unseeded randomness, no hidden manual steps.
