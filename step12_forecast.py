import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

FEATURED_PATH = "outputs/featured_dataset.csv"
MODEL_DIR = "outputs/models"
OUT_PATH = "outputs/forecast.csv"

BATCH_CADENCE_DAYS = 7  # matches the observed near-weekly operating pattern
HORIZONS = [7, 30, 60, 90]

FEATURE_COLS = [
    "day_of_week", "week_of_year", "month", "day_of_year", "is_weekend",
    "avg_lead_time", "median_lead_time",
    "participant_count_lag_1", "participant_count_lag_2", "participant_count_lag_4",
    "participant_count_rolling_mean_4",
    "class_revenue_lag_1", "class_revenue_lag_2", "class_revenue_lag_4",
    "class_revenue_rolling_mean_4", "season_is_dry",
]


def get_season(month):
    return "dry" if month in (11, 12, 1, 2, 3, 4) else "wet"


df = pd.read_csv(FEATURED_PATH, parse_dates=["batch_date"])
# Forecast from the last reliably-dated batch, not the extrapolated tail -
# generating forecasts on top of an already-guessed date would compound
# uncertainty on uncertainty.
eligible = df[~df["date_source"].str.contains("extrapolated", na=False)].sort_values("batch_date").reset_index(drop=True)
last_known_date = eligible["batch_date"].max()

# ---- Historical thresholds for classification (data-driven, needs client
#      validation per the original methodology - these are a reasonable
#      starting point, not a final business decision) ----
p33, p67 = eligible["participant_count"].quantile([1/3, 2/3])


def classify_demand(participants):
    if participants <= p33:
        return "Low"
    elif participants <= p67:
        return "Medium"
    return "High"


monthly_avg = eligible.groupby(eligible["batch_date"].dt.month)["participant_count"].mean()
month_tercile = monthly_avg.quantile([1/3, 2/3])


def classify_season_period(month):
    avg = monthly_avg.get(month, monthly_avg.mean())
    if avg <= month_tercile.iloc[0]:
        return "Off-Peak"
    elif avg <= month_tercile.iloc[1]:
        return "Shoulder"
    return "Peak"


participant_model = joblib.load(f"{MODEL_DIR}/participant_count_model.joblib")
revenue_model = joblib.load(f"{MODEL_DIR}/class_revenue_model.joblib")

# Seed the recursive lag/rolling window with the last 4 known (non-extrapolated) batches
history = eligible[["batch_date", "participant_count", "class_revenue"]].tail(4).copy()
# booking lead time has no future equivalent (we don't know future booking behavior
# yet) - carry forward the historical average as a neutral placeholder
avg_lead = eligible["avg_lead_time"].mean()
median_lead = eligible["median_lead_time"].mean()

forecast_rows = []
current_date = last_known_date
n_steps = (max(HORIZONS) // BATCH_CADENCE_DAYS) + 1

for step in range(n_steps):
    current_date = current_date + timedelta(days=BATCH_CADENCE_DAYS)

    recent_participants = history["participant_count"].tolist()
    recent_revenue = history["class_revenue"].tolist()

    row = {
        "day_of_week": current_date.dayofweek,
        "week_of_year": current_date.isocalendar()[1],
        "month": current_date.month,
        "day_of_year": current_date.dayofyear,
        "is_weekend": int(current_date.dayofweek in (5, 6)),
        "avg_lead_time": avg_lead,
        "median_lead_time": median_lead,
        "participant_count_lag_1": recent_participants[-1],
        "participant_count_lag_2": recent_participants[-2] if len(recent_participants) >= 2 else np.nan,
        "participant_count_lag_4": recent_participants[-4] if len(recent_participants) >= 4 else np.nan,
        "participant_count_rolling_mean_4": np.mean(recent_participants[-4:]),
        "class_revenue_lag_1": recent_revenue[-1],
        "class_revenue_lag_2": recent_revenue[-2] if len(recent_revenue) >= 2 else np.nan,
        "class_revenue_lag_4": recent_revenue[-4] if len(recent_revenue) >= 4 else np.nan,
        "class_revenue_rolling_mean_4": np.mean(recent_revenue[-4:]),
        "season_is_dry": int(get_season(current_date.month) == "dry"),
    }
    X_step = pd.DataFrame([row])[FEATURE_COLS]

    pred_participants = max(0, float(participant_model.predict(X_step)[0]))
    pred_revenue = max(0, float(revenue_model.predict(X_step)[0]))

    forecast_rows.append({
        "forecast_date": current_date.date(),
        "days_ahead": (current_date - last_known_date).days,
        "predicted_participants": round(pred_participants, 1),
        "predicted_revenue_php": round(pred_revenue, 0),
        "demand_level": classify_demand(pred_participants),
        "season_period": classify_season_period(current_date.month),
        "instructors_needed": int(np.ceil(pred_participants / 4)),
    })

    # feed this prediction into the next step's lag window - this is the
    # recursive step; errors here compound over the horizon, which is expected
    # and should be reported, not hidden
    history = pd.concat([history, pd.DataFrame([{
        "batch_date": current_date, "participant_count": pred_participants, "class_revenue": pred_revenue
    }])], ignore_index=True).tail(4)

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(OUT_PATH, index=False)

print(f"Forecasting from {last_known_date.date()} forward, {n_steps} synthetic batches at "
      f"{BATCH_CADENCE_DAYS}-day cadence\n")
print(forecast_df.to_string(index=False))

print(f"\nDemand-level thresholds: Low <= {p33:.1f}, Medium <= {p67:.1f}, High > {p67:.1f} participants "
      f"(needs client validation)")

print("\nHorizon summaries:")
for h in HORIZONS:
    window = forecast_df[forecast_df["days_ahead"] <= h]
    print(f"  {h}-day: {len(window)} batches, "
          f"{window['predicted_participants'].sum():.0f} participants, "
          f"{window['predicted_revenue_php'].sum():,.0f} PHP revenue, "
          f"peak instructor need {window['instructors_needed'].max() if len(window) else 0}")

print(f"\nSaved -> {OUT_PATH}")