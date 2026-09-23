# IDX-ML-Lab — 8-Week Daily Learning Map

**Goal:** Review Python engineering, classical Machine Learning, and Deep Learning fundamentals by building one cohesive, portfolio-quality project — an end-to-end signal-prediction & backtesting pipeline for Indonesian (IDX) stocks — with a clean, daily-committed GitHub history.

**Assumed baseline:** Python fundamentals, ML fundamentals, DL fundamentals already covered in coursework. This map is a *review + integration* track, not an intro course.

**Cadence:** ~6 working days/week (rest one day), ~30–60 min/day. 8 weeks ≈ your 1–2 month window — stop after Week 6 if you only have ~6 weeks; Weeks 7–8 (productization) are the stretch/polish phase.

**Disclaimer:** This project produces *educational* trading signals for skill review only. It is not financial advice and should not be used with real capital without independent validation.

---

## How to use this map

1. One GitHub repo, e.g. `idx-ml-lab`. One task = one commit (or a few small commits), with a descriptive message (`feat:`, `fix:`, `test:`, `docs:` prefixes recommended).
2. At the end of each week, write 3–5 lines in `docs/journal.md` (what you built, what was hard, what you'd do differently). This journal *is* the proof-of-consistency the rest of your README will point to.
3. Tag two milestones: `v0.1-classical-ml` (end of Week 4) and `v0.2-deep-learning` (end of Week 6), plus `v1.0` at the very end.
4. If a day's task takes longer than planned, don't skip the commit — commit what you have and carry the rest to a buffer slot (Day 6 of each week is lighter by design).

---

## Week 1 — Setup & Data Engineering

| Day | Task | Output |
|---|---|---|
| 1 | Init repo: `src/`, `notebooks/`, `tests/`, `data/`, `docs/` folders; set up venv/poetry; skeleton README | First commit |
| 2 | `DataLoader` class using `yfinance` to pull OHLCV for 5 LQ45 tickers (e.g. BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, UNVR.JK), cache as parquet | `src/idxlab/data.py` |
| 3 | Cleaning module: missing values, trading-calendar alignment, split handling; type hints + docstrings | `src/idxlab/clean.py` |
| 4 | `pytest` suite for loader + cleaning | `tests/test_data.py` green |
| 5 | Config via YAML/dotenv (tickers, date ranges); `logging` setup; refactor into installable package | `idxlab/` package |
| 6 | EDA notebook (price/volume trends, missing-data report); Week 1 journal entry | `notebooks/01_eda.ipynb` |

## Week 2 — Feature Engineering & Labeling

| Day | Task | Output |
|---|---|---|
| 7 | Indicators: SMA, EMA, RSI, MACD, Bollinger Bands | `src/idxlab/indicators.py` |
| 8 | More indicators (ATR, OBV, momentum) + a `FeaturePipeline` class | `src/idxlab/features.py` |
| 9 | Labeling module: next-day up/down or N-day forward return, careful about look-ahead bias | `src/idxlab/labels.py` |
| 10 | Time-based train/val/test split (no shuffling — walk-forward) | `src/idxlab/split.py` |
| 11 | Tests for features + labeling; type hints project-wide | `tests/test_features.py` |
| 12 | Docstrings pass, simple architecture sketch, Week 2 journal | `docs/architecture.md` |

## Week 3 — Classical ML I

| Day | Task | Output |
|---|---|---|
| 13 | Logistic Regression baseline + evaluation harness (accuracy, precision/recall, ROC-AUC) | `notebooks/02_baseline.ipynb` |
| 14 | Random Forest + XGBoost/LightGBM, compared to baseline | Model comparison table |
| 15 | Feature importance (SHAP or built-in) | `docs/feature_importance.md` |
| 16 | Hyperparameter tuning with `TimeSeriesSplit` + `Optuna`/`GridSearchCV` | Tuned model |
| 17 | Check class imbalance, evaluate mitigation impact | Notes in notebook |
| 18 | Model comparison report; Week 3 journal | `docs/results_classical.md` |

## Week 4 — Classical ML II: Backtesting

| Day | Task | Output |
|---|---|---|
| 19 | Simple backtest engine: signal → simulated trades w/ transaction costs | `src/idxlab/backtest.py` |
| 20 | Metrics: cumulative return, Sharpe, max drawdown, win rate; equity-curve plot | `notebooks/03_backtest.ipynb` |
| 21 | Compare strategy vs. buy-and-hold across all 5 tickers | Comparison chart |
| 22 | Walk-forward rolling-window backtest (avoid lookahead illusions) | Updated backtest |
| 23 | Error analysis: worst predictions, regime dependency | `docs/error_analysis.md` |
| 24 | **Milestone:** `docs/midpoint_review.md`, tag `v0.1-classical-ml` | Git tag |

## Week 5 — Deep Learning I

| Day | Task | Output |
|---|---|---|
| 25 | Sliding-window sequence dataset builder (TF/Keras `Dataset` or PyTorch `Dataset`) | `src/idxlab/sequences.py` |
| 26 | MLP baseline on flattened windows | `notebooks/04_dl_baseline.ipynb` |
| 27 | LSTM model — build, train, tune sequence length/units | LSTM checkpoint |
| 28 | GRU variant + dropout/early stopping; simple experiment log (CSV is fine) | `docs/experiments.csv` |
| 29 | Learning-curve diagnostics; fix under/overfitting | Updated model |
| 30 | Week 5 journal, save best DL checkpoint | `models/best_lstm.keras` |

## Week 6 — Deep Learning II

| Day | Task | Output |
|---|---|---|
| 31 | Multi-ticker shared model vs. per-ticker models — which generalizes better? | Comparison notes |
| 32 | *(Stretch)* Attention layer or lightweight Transformer block | Optional model variant |
| 33 | Ensemble classical + DL (averaging/stacking), backtest it | `src/idxlab/ensemble.py` |
| 34 | Out-of-sample robustness check on a volatile period | `docs/robustness.md` |
| 35 | *(Optional)* DL interpretability (attention weights / integrated gradients) | Notebook cell |
| 36 | **Milestone:** Week 6 journal, tag `v0.2-deep-learning` | Git tag |

## Week 7 — Productization I

| Day | Task | Output |
|---|---|---|
| 37 | Package pipeline as installable module (`pyproject.toml`) | Installable `idxlab` |
| 38 | FastAPI `/predict` endpoint (ticker → signal + confidence) | `src/api/main.py` |
| 39 | Pydantic request validation, error handling, API tests | `tests/test_api.py` |
| 40 | Dockerize (`Dockerfile`, `docker-compose.yml`) | Working container |
| 41 | GitHub Actions CI: pytest + lint (ruff/flake8) on push | `.github/workflows/ci.yml` |
| 42 | Week 7 journal, confirm CI badge is green | — |

## Week 8 — Productization II & Polish

| Day | Task | Output |
|---|---|---|
| 43 | Streamlit dashboard: pick ticker → price chart + signal + equity curve | `app/dashboard.py` |
| 44 | Polish UX, cache expensive calls, document how to run | — |
| 45 | Final README pass: architecture diagram, results table, run instructions, disclaimer | Updated `README.md` |
| 46 | Demo GIF/screenshots, badges (build, license, Python version) | `docs/demo.gif` |
| 47 | Reproducibility check: fresh clone → follow README → confirm it actually runs | — |
| 48 | **Final:** `docs/final_review.md` retrospective, tag `v1.0` | Git tag |

---

## Stretch ideas (if you finish early)
- Swap `yfinance` for a broader universe (IDX30, all LQ45) and check if the pipeline scales cleanly.
- Add MLflow (or even just structured JSON logs) for proper experiment tracking.
- Write a short blog-style post in `docs/` explaining the CMEM/ALNS-style rigor you already use in your thesis, applied here to model validation discipline (nice narrative bridge between your two DSAI projects).
