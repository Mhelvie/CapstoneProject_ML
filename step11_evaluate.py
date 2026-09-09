import pandas as pd
import numpy as np
import joblib

SPLIT_PATH = "outputs/split_dataset.csv"
MODEL_DIR = "outputs/models"

df = pd.read_csv(SPLIT_PATH, parse_dates=["batch_date"])
test = df[df["split"] == "test"].sort_values("batch_date").reset_index(drop=True)

FEATURE_COLS = [
    "day_of_week", "week_of_year", "month", "day_of_year", "is_weekend",
    "avg_lead_time", "median_lead_time",
    "participant_count_lag_1", "participant_count_lag_2", "participant_count_lag_4",
    "participant_count_rolling_mean_4",
    "class_revenue_lag_1", "class_revenue_lag_2", "class_revenue_lag_4",
    "class_revenue_rolling_mean_4", "season_is_dry",
]
test["season_is_dry"] = (test["season"] == "dry").astype(int)
X_test = test[FEATURE_COLS]

TARGETS = ["participant_count", "class_revenue"]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    # MAPE undefined at y_true == 0 - exclude those rows from the percentage
    # calculation rather than dividing by zero, but keep them in MAE/RMSE/MBE.
    nonzero = y_true != 0
    mape = np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100 if nonzero.any() else float("nan")
    wape = np.sum(np.abs(errors)) / np.sum(y_true) * 100 if np.sum(y_true) != 0 else float("nan")
    mbe = np.mean(errors)  # positive = overforecasting, negative = underforecasting
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "MBE": mbe,
            "n_excluded_from_mape": int((~nonzero).sum())}


print(f"Test set: {len(test)} batches, {test['batch_date'].min().date()} -> {test['batch_date'].max().date()}\n")

small_batches = test[test["participant_count"] < 5][["batch_id", "batch_date", "participant_count", "class_revenue"]]
if len(small_batches):
    print("⚠ Unusually small batches in the test set (MAPE is highly sensitive to these -"
          " worth checking whether they're genuine single/small bookings or incomplete records):")
    print(small_batches.to_string(index=False))
    print()

results = {}
for target in TARGETS:
    model = joblib.load(f"{MODEL_DIR}/{target}_model.joblib")
    y_true = test[target]
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_true, y_pred)
    results[target] = metrics

    print(f"{target}:")
    print(f"  MAE:  {metrics['MAE']:,.2f}")
    print(f"  RMSE: {metrics['RMSE']:,.2f}")
    print(f"  MAPE: {metrics['MAPE']:,.1f}%"
          + (f"  ({metrics['n_excluded_from_mape']} zero-value row(s) excluded)" if metrics["n_excluded_from_mape"] else ""))
    print(f"  WAPE: {metrics['WAPE']:,.1f}%  (weighted by actual size - much less skewed by small-batch outliers than MAPE)")
    print(f"  MBE:  {metrics['MBE']:,.2f}  "
          f"({'overforecasting' if metrics['MBE'] > 0 else 'underforecasting'} on average)")
    print()

# Per-batch prediction table, useful for spotting which specific batches the
# model got most wrong (e.g. an unusual event batch it had no way to anticipate).
detail = test[["batch_id", "batch_date"]].copy()
for target in TARGETS:
    model = joblib.load(f"{MODEL_DIR}/{target}_model.joblib")
    detail[f"{target}_actual"] = test[target].values
    detail[f"{target}_predicted"] = model.predict(X_test)
    detail[f"{target}_error"] = detail[f"{target}_predicted"] - detail[f"{target}_actual"]

detail.to_csv("outputs/test_predictions.csv", index=False)
print("Per-batch predictions saved -> outputs/test_predictions.csv")