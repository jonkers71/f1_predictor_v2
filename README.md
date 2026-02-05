# F1 Predictor v2 🏎️💨

A modern, high-performance Formula 1 prediction engine and accuracy analysis dashboard. This project uses real-world telemetry and sector data to predict Qualifying and Race outcomes with high precision.

## 🚀 Key Features

- **Weighted Prediction Engine**: Uses a 75/25 weighted model combining raw pace with constructor strength.
- **Ideal Lap Analysis**: Calculates theoretical best laps by summing Sector 1, 2, and 3 timings from practice sessions.
- **Reliability Index**: Penalizes high-variance performances and accounts for rookie outliers.
- **Accuracy Analysis (Backtesting)**: Deep-dive tool to compare predicted vs actual historical results for any race from 2023-2026.
- **Smart Navigation**: Context-aware dropdowns that dynamically adapt to Sprint vs. Standard weekends with red "🔴 S" indicators.
- **Automated Data Engine**: Integrated with `FastF1` for seamless, automated telemetry fetching and caching.

## 🏗️ Architecture

- **Frontend**: Next.js 15 (App Router), Vanilla CSS, Tailwind, Responsive "Race Command Center" UI.
- **Backend**: FastAPI (Python), Numpy, Pandas.
- **Data Engine**: `FastF1` for precise telemetry and session data.
- **ML Models**: Weighted scoring logic based on performance deltas and consistency metrics.

## 🛠️ Setup & Installation

### 1. Backend (Python)
Ensure you have Python 3.9+ installed.

```bash
cd backend
pip install fastapi uvicorn fastf1 pandas numpy xgboost joblib tenacity
python start_f1.py
```
*Note: The first run will create a `cache/` directory. Initial data downloads for historical years may take several minutes.*

### 2. Frontend (Next.js)
Ensure you have Node.js 18+ installed.

```bash
cd frontend
npm install
npm run dev
```
The app will be available at `http://localhost:3000`.

## 📊 Accuracy Performance

In backtesting against the 2024/2025 seasons:
- **Mean Absolute Error (MAE)**: ~0.22s to 0.28s for Qualifying.
- **Top 5 Correlation**: High accuracy in predicting front-row starts based on FP3 Ideal Lap data.

## 📝 License
MIT
