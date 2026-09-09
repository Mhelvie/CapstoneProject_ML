import os
import requests
import pandas as pd
import numpy as np

# Load config from environment variables (with local fallbacks)
LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://127.0.0.1:8000/api/v1/ml")
ML_TOKEN = os.getenv("ML_TOKEN", "cfml_live_8eef173d6b5670cab3ec93d7ae53736a4128e079484c718819f20625")

headers = {
    "Authorization": f"Bearer {ML_TOKEN}",
    "Accept": "application/json"
}

# ==========================================
# STEP 1: FETCH FRESH DATA FROM LARAVEL
# ==========================================
print("Fetching fresh training data from Laravel...")
try:
    response = requests.get(f"{LARAVEL_API_URL}/training-data", headers=headers, timeout=15)
    response.raise_for_status()
    records = response.json().get("data", [])
    df_fresh = pd.DataFrame(records)
    print(f"-> Successfully retrieved {len(df_fresh)} completed batch records.")
except Exception as e:
    print(f"Warning: Could not fetch from Laravel ({e}). Falling back to local dataset if available.")
    df_fresh = pd.DataFrame()

# ==========================================
# STEP 2: RETRAIN MODELS & GENERATE FORECAST
# ==========================================
# (Your model retraining & feature engineering code goes here)
# ...
# Output: df_forecast (DataFrame with columns: forecast_date, days_ahead, 
#         predicted_participants, predicted_revenue_php, demand_level, season_period, instructors_needed)

# Save local CSV backup
df_forecast.to_csv("outputs/forecast.csv", index=False)

# ==========================================
# STEP 3: COMPUTE HORIZONS & PUSH TO LARAVEL
# ==========================================
# Compute dynamic horizon summaries directly from df_forecast
def get_horizon_summary(df, max_days):
    subset = df[df["days_ahead"] <= max_days]
    if subset.empty:
        return {"batches": 0, "participants": 0, "revenue": 0, "peak_instructors": 0}
    return {
        "batches": int(len(subset)),
        "participants": int(round(subset["predicted_participants"].sum())),
        "revenue": float(round(subset["predicted_revenue_php"].sum(), 2)),
        "peak_instructors": int(subset["instructors_needed"].max())
    }

horizon_summaries = {
    "7_day": get_horizon_summary(df_forecast, 7),
    "30_day": get_horizon_summary(df_forecast, 30),
    "60_day": get_horizon_summary(df_forecast, 60),
    "90_day": get_horizon_summary(df_forecast, 90),
}

forecast_payload = {
    "horizon_summaries": horizon_summaries,
    "forecasts": df_forecast.to_dict(orient="records")
}

print("Syncing 90-day forecast to Laravel database...")
try:
    sync_response = requests.post(
        f"{LARAVEL_API_URL}/sync-forecast",
        json=forecast_payload,
        headers=headers,
        timeout=15
    )
    sync_response.raise_for_status()
    print("-> Sync complete:", sync_response.json().get("message"))
except Exception as e:
    print(f"-> Error syncing forecast to Laravel: {e}")