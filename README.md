# Turkish Economic Indicators Forecasting System

> **Research project supported by TÜBİTAK (The Scientific and Technological Research Council of Turkey)**

A production-grade machine learning system for forecasting key Turkish economic indicators — USD/TRY exchange rate, inflation (CPI), and central bank policy interest rate — using hybrid time-series models, NLP-driven news sentiment analysis, and real-time data pipelines.

---

## Forecast Accuracy

| Indicator | Model | MAPE |
|-----------|-------|------|
| USD/TRY Exchange Rate | Hybrid ARIMA-SVR | **3.45%** |
| Inflation (CPI) | ARIMA + Sentiment | **9.82%** |
| Policy Interest Rate | ARIMA + Sentiment | **9.52%** |

All models achieve **< 10% MAPE** validated through walk-forward backtesting and hold-out testing.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  React / TypeScript Dashboard             │
│             (Vite + Tailwind + Lightweight Charts)        │
├──────────────────────────────────────────────────────────┤
│                      Flask REST API                       │
├───────────┬───────────┬───────────────┬──────────────────┤
│   Data    │  Models   │     NLP       │   Validation     │
│  Pipeline │  Engine   │  Sentiment    │   Framework      │
├───────────┼───────────┼───────────────┼──────────────────┤
│ TCMB API  │ ARIMA     │ Claude AI     │ Walk-Forward     │
│ NewsAPI   │ Prophet   │ TextBlob      │ Cross-Validation │
│ Web       │ XGBoost   │ Turkish NLP   │ Backtesting      │
│ Scraping  │ LightGBM  │ Keyword Dict  │ Hold-out         │
│ CSV Data  │ LSTM      │               │ Stress Testing   │
│           │ SVR       │               │                  │
├───────────┴───────────┴───────────────┴──────────────────┤
│                        MongoDB                            │
└──────────────────────────────────────────────────────────┘
```

---

## Key Features

### Multi-Model Forecasting Engine
- **ARIMA** with automatic parameter optimization via pmdarima
- **Hybrid ARIMA-SVR** methodology combining linear and non-linear patterns
- **Prophet**, **XGBoost**, **LightGBM**, and **LSTM** neural networks
- Ensemble methods with weighted model averaging

### NLP-Driven Sentiment Analysis
- Real-time Turkish economic news collection via NewsAPI
- Claude AI integration for contextual sentiment scoring
- Sentiment multiplier applied to base ARIMA forecasts
- Turkish-specific keyword dictionaries for economic context

### Automated Data Pipeline
- TCMB EVDS API (Turkish Central Bank) for official economic data
- Web scraping for historical exchange rate data
- Multi-source data integration and preprocessing
- Scheduled data collection with automatic model retraining

### Validation Framework
- Walk-forward and rolling-window backtesting
- Time-series adapted K-fold cross-validation
- MAPE, MAE, RMSE metrics with configurable accuracy thresholds
- Regime-aware data windowing and Winsorization for outlier handling

### Web Dashboard
- Interactive forecast charts with TradingView Lightweight Charts
- Multi-indicator monitoring with real-time updates
- News sentiment summary panel
- Responsive design (React + TypeScript + Tailwind CSS)

### Reporting & Export
- Automated PDF report generation (ReportLab)
- Excel export with formatted tables (openpyxl)
- Comparative model performance analysis

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React, TypeScript, Vite, Tailwind CSS, Lightweight Charts |
| **Backend** | Python 3.10+, Flask, Gunicorn |
| **ML / Forecasting** | statsmodels, pmdarima, scikit-learn, Prophet, XGBoost, LightGBM, TensorFlow/Keras |
| **NLP** | Anthropic Claude API, TextBlob |
| **Database** | MongoDB (PyMongo) |
| **Data Sources** | TCMB EVDS API, NewsAPI, Web Scraping |
| **Testing** | pytest, pytest-cov |
| **CI/CD** | GitHub Actions |

---

## Project Structure

```
├── app.py                              # Flask REST API server
├── main.py                             # CLI dashboard & admin tool
├── config/
│   └── config.py                       # Centralized configuration
├── src/
│   ├── core/
│   │   └── economic_analyzer.py        # Multi-indicator coordinator
│   ├── models/
│   │   ├── arima_model.py              # Base ARIMA forecaster
│   │   ├── inflation_forecaster.py     # Inflation-tuned model
│   │   ├── interest_rate_forecaster.py # Interest rate model
│   │   ├── advanced_forecaster.py      # Prophet, XGBoost, LSTM
│   │   ├── svr_model.py               # Support Vector Regression
│   │   └── model_validator.py          # Validation framework
│   ├── services/
│   │   ├── news_api_service.py         # News + Claude AI integration
│   │   └── multi_indicator_service.py  # Cross-indicator analysis
│   ├── data_collection/
│   │   ├── tcmb_data_collector.py      # Central Bank API client
│   │   ├── turkish_financial_scraper.py
│   │   ├── csv_processor.py
│   │   └── data_integration.py         # Multi-source pipeline
│   ├── nlp_analysis/
│   │   └── sentiment_analyzer.py       # Turkish sentiment engine
│   ├── visualization/
│   │   └── forecast_visualizer.py      # Matplotlib charts
│   ├── analysis/
│   │   └── comparative_analyzer.py     # Model comparison
│   ├── systems/
│   │   └── autonomous_forecaster.py    # End-to-end pipeline
│   ├── testing/
│   │   ├── backtesting.py              # Walk-forward testing
│   │   └── stress_tester.py            # Model stress tests
│   └── utils/
│       ├── mongodb_manager.py          # Database operations
│       ├── accuracy_calculator.py      # MAPE/MAE/RMSE
│       ├── data_preprocessor.py        # Data cleaning
│       ├── report_exporter.py          # PDF/Excel generation
│       └── ...
├── frontend/                           # React/TypeScript UI
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   │       ├── ChartPanel.tsx          # Interactive forecast charts
│   │       ├── IndicatorCard.tsx       # Indicator display cards
│   │       ├── NewsSummary.tsx         # Sentiment summary
│   │       └── ...
│   └── vite.config.ts
└── tests/                              # Test suite
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB 6.0+

### Installation

```bash
git clone https://github.com/<your-username>/turkish-economic-forecasting.git
cd turkish-economic-forecasting

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### Configuration

Create a `.env` file in the project root:

```env
NEWSAPI_KEY=your_newsapi_key
CLAUDE_API_KEY=your_claude_api_key
TCMB_API_KEY=your_tcmb_api_key

MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=economic_data
```

### Running

```bash
# Backend API
python app.py

# Frontend dev server (separate terminal)
cd frontend && npm run dev

# Or use the CLI dashboard
python main.py
```

Dashboard: `http://localhost:5000` | Frontend dev: `http://localhost:5173`

---

## Methodology

1. **Data Collection** — Historical data from TCMB, web scraping, and curated CSV datasets
2. **Preprocessing** — Regime-aware windowing (post-2021), log transformation, Winsorization
3. **Model Training** — Auto-ARIMA optimization per indicator, hybrid ARIMA-SVR fitting
4. **Sentiment Analysis** — Real-time news scoring via Claude AI with Turkish NLP
5. **Forecast Generation** — Base forecast adjusted by news sentiment multiplier
6. **Validation** — Walk-forward backtesting, hold-out testing, cross-validation

---

## Testing

```bash
pytest                                  # Run all tests
pytest --cov=src --cov-report=html      # With coverage report
python test_all_mape.py                 # Validate MAPE targets
```

---

## Acknowledgments

This research was supported by [TÜBİTAK](https://www.tubitak.gov.tr/) (The Scientific and Technological Research Council of Turkey) under their competitive research grant program.

---

## License

All rights reserved. This project was developed as part of a TÜBİTAK-funded research initiative.
