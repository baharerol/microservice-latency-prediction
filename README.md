# API Performance Management - Predictive Analytics & Explainable AI

A machine learning system that predicts API response times before slowdowns happen, classifies system load into risk levels, and explains its predictions with SHAP. Undergraduate graduation project, Software Engineering, Samsun University (supervisor: Prof. Dr. Zafer Cömert).

> **Status:** active development. The current version is a prototype validated on synthetic data with a local Streamlit dashboard. See [Roadmap](#roadmap) for the in-progress live-API and class-imbalance work.

## Motivation

Most API monitoring is reactive, problems are detected only after they occur. This project aims to predict response-time degradation in advance, turn those predictions into operational risk categories, and make the model's reasoning interpretable so an operator knows *why* a slowdown is coming.

## Approach

1. **Synthetic data** - a FastAPI simulation generates ~29,000 request records across 5 load scenarios (low, medium, high, spike, mixed) and 4 endpoints, with response time driven by concurrent requests, RPS, payload size and random spikes. Real production traffic was out of reach (privacy / access constraints), so controlled synthetic generation was used.
2. **Feature engineering** - lag features (previous 1–3 response times) and rolling statistics (mean over 5/10, std over 5), all computed with `shift()` to avoid data leakage.
3. **Models** - Linear Regression, Random Forest and LSTM, each in a baseline and a feature-engineered (enhanced) configuration. Temporal (chronological 80/20) split, no shuffling.
4. **Explainability** - SHAP values to rank which features drive each prediction.
5. **Risk classification** - percentile-based thresholds (P95 / P99 style) mapped to multi-class risk levels.

## Results

**Response-time prediction (regression)**
- Best model: Random Forest (enhanced) - **MAE 88.17**, **R² 0.410**.
- Feature engineering raised R² from **0.02 → 0.41**, confirming the models could not capture temporal dependencies without lag/rolling features.
- After feature engineering, RF (R² 0.410) and LSTM (R² 0.399) performed almost the same - the lag features largely replicate what LSTM's recurrent architecture learns on its own.

**Explainability**
- SHAP identified `response_time_lag_1` (the immediately preceding response time) as the dominant predictor.

**Risk classification**
- Overall accuracy was 88.5%, but this is driven by the majority (low-risk) class. **Critical-class recall is only 2.4%** — i.e. the model currently misses almost all genuinely critical cases. This is a class-imbalance failure and is the main open problem; addressing it is the first item on the roadmap below.

## Limitations

- **Synthetic data** - the model learns from a self-designed generation process, so results validate the *method*, not real-world performance.
- **Class imbalance** - critical events are rare, which collapses Critical-class recall (see above).
- **Modest R²** - the regression explains ~41% of variance; useful as a prototype, not production-ready.

## Roadmap

1. **Fix Critical-class recall** - class weights / resampling (e.g. SMOTE) and decision-threshold tuning, so the system actually catches the high-risk cases it exists to warn about.
2. **Serve the model as a live FastAPI service** - a `/predict` endpoint returning predicted response time, risk class and a SHAP explanation, deployed publicly, replacing the static dashboard with a real prediction service.

## Tech stack

- **Language:** Python
- **API / simulation:** FastAPI
- **ML:** scikit-learn (Linear Regression, Random Forest), TensorFlow/Keras (LSTM)
- **Explainability:** SHAP
- **Data / viz:** pandas, Streamlit


## Dashboard

Interactive Streamlit dashboard (overview, prediction & risk, SHAP analysis, what-if pages). Runs locally - not deployed yet.


## Setup

```bash
git clone https://github.com/baharerol/microservice-latency-prediction.git
cd microservice-latency-prediction
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

<!-- TODO: fill in the real repo name and dashboard entry file. -->

## Author

Bahar Erol
