# IDX-ML-Lab

*A daily-built, end-to-end machine learning pipeline for Indonesian (IDX) stock signal prediction & backtesting — a Python/ML/DL skills-review project.*

![Python](https://img.shields.io/badge/python-3.11-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-in--progress-yellow)

> ⚠️ **Disclaimer:** Built for learning and portfolio purposes. Nothing here is financial advice — signals are not validated for real trading.

---

## Why this project

I'm a Data Science & AI student currently interning as a Web Developer, working mainly in Go rather than Python/ML day-to-day. This project is my way of keeping my core DSAI skills — Python engineering, classical ML, and deep learning — sharp, while producing something I actually care about (I'm interested in the Indonesian stock market). It's built in daily increments over ~8 weeks; see [`docs/journal.md`](docs/journal.md) for the week-by-week log and [`LEARNING_MAP.md`](LEARNING_MAP.md) for the full day-by-day plan.

## What it does

1. **Data** — pulls OHLCV data for a set of LQ45 tickers via `yfinance`, cleans and caches it.
2. **Features** — computes technical indicators (SMA/EMA/RSI/MACD/Bollinger/ATR/OBV) and builds walk-forward-safe train/val/test splits.
3. **Classical ML** — Logistic Regression, Random Forest, and XGBoost models predicting next-day price direction, tuned with time-series cross-validation.
4. **Deep Learning** — MLP → LSTM → GRU sequence models on the same task, plus a classical+DL ensemble.
5. **Backtesting** — turns model signals into a simulated strategy and reports cumulative return, Sharpe ratio, max drawdown, and win rate vs. buy-and-hold.
6. **Serving** — a FastAPI `/predict` endpoint and a small Streamlit dashboard, containerized with Docker and checked by GitHub Actions CI.

## Tech stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `yfinance` · `FastAPI` · `Streamlit` · `Docker` · `GitHub Actions` · `pytest`

## Project structure

```
idx-ml-lab/
├── src/
│   ├── idxlab/          # core package: data, features, models, backtest
│   └── api/              # FastAPI service
├── app/                   # Streamlit dashboard
├── notebooks/             # EDA & experiment notebooks
├── tests/                 # pytest suite
├── docs/                  # journal, architecture notes, results, retrospective
├── models/                # saved checkpoints
├── LEARNING_MAP.md        # the 8-week day-by-day plan this repo follows
└── README.md
```

## Results

*(fill in as Weeks 3–6 complete)*

| Model | Accuracy | Sharpe (backtest) | Max Drawdown |
|---|---|---|---|
| Logistic Regression (baseline) | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| LSTM | — | — | — |
| Ensemble | — | — | — |

## How to run

```bash
git clone https://github.com/<your-username>/idx-ml-lab.git
cd idx-ml-lab
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# run the pipeline
python -m idxlab.pipeline --tickers BBCA.JK BBRI.JK TLKM.JK

# serve the API
uvicorn src.api.main:app --reload

# launch the dashboard
streamlit run app/dashboard.py
```

## Progress

- [ ] Week 1–2 — Data engineering & feature pipeline
- [ ] Week 3–4 — Classical ML & backtesting (`v0.1-classical-ml`)
- [ ] Week 5–6 — Deep learning models (`v0.2-deep-learning`)
- [ ] Week 7–8 — API, dashboard, CI, polish (`v1.0`)

## Lessons learned

*(populated from `docs/journal.md` and `docs/final_review.md` at the end of the 8 weeks)*

## License

MIT — see [LICENSE](LICENSE).
