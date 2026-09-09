import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import joblib

SPLIT_PATH = "outputs/split_dataset.csv"
MODEL_DIR = "outputs/models"

import os
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(SPLIT_PATH, parse_dates=["batch_date"])

# Only rows actually assigned to train/validation participate in fitting.
# (split == None/NaN rows - the extrapolated + excluded-window batches - are
# left out entirely, same as Step 9 intended.)
usable = df[df["split"].isin(["train", "validation"])].sort_values("batch_date").reset_index(drop=True)

FEATURE_COLS = [
    "day_of_week", "week_of_year", "month", "day_of_year", "is_weekend",
    "avg_lead_time", "median_lead_time",
    "participant_count_lag_1", "participant_count_lag_2", "participant_count_lag_4",
    "participant_count_rolling_mean_4",
    "class_revenue_lag_1", "class_revenue_lag_2", "class_revenue_lag_4",
    "class_revenue_rolling_mean_4",
]
# season is categorical text ("wet"/"dry") - encode it numerically for XGBoost
usable["season_is_dry"] = (usable["season"] == "dry").astype(int)
FEATURE_COLS.append("season_is_dry")

TARGETS = ["participant_count", "class_revenue"]

X = usable[FEATURE_COLS]
# XGBoost handles NaN natively (missing=np.nan is the default), so the rows
# with no lag_4 history yet, or no booking-lead-time data, do NOT need to be
# dropped - leaving NaN in place lets the model learn around missing values
# rather than losing those rows entirely.

# ---- Use a PredefinedSplit so GridSearchCV tunes hyperparameters against the
#      validation set specifically, not a random k-fold - which would violate
#      chronological ordering by validating on rows that come before some
#      training rows. -1 marks train rows, 0 marks the validation fold. ----
test_fold = usable["split"].map({"train": -1, "validation": 0}).values
ps = PredefinedSplit(test_fold)

PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
}

models = {}
for target in TARGETS:
    y = usable[target]
    grid = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid=PARAM_GRID,
        cv=ps,
        scoring="neg_mean_absolute_error",
        refit=True,
    )
    grid.fit(X, y)
    models[target] = grid.best_estimator_

    joblib.dump(grid.best_estimator_, f"{MODEL_DIR}/{target}_model.joblib")

    print(f"\n{target}:")
    print(f"  best params: {grid.best_params_}")
    print(f"  best validation MAE: {-grid.best_score_:,.1f}")

print(f"\nModels saved to {MODEL_DIR}/")
print("Feature columns used (same order needed at prediction time):")
print(FEATURE_COLS)