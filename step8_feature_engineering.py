import pandas as pd

BATCH_PATH = "outputs/batch_level_dataset.csv"
TIER1_PATH = "outputs/tier1_cleaned.csv"
OUT_PATH = "outputs/featured_dataset.csv"


def get_season(month):
    # Philippines: roughly two seasons - dry (Nov-Apr) and wet (May-Oct).
    # Adjust if the client defines peak/off-peak differently for diving demand.
    return "dry" if month in (11, 12, 1, 2, 3, 4) else "wet"


batch = pd.read_csv(BATCH_PATH, parse_dates=["batch_date"])
batch = batch.sort_values("batch_date").reset_index(drop=True)

# ---- 1. Temporal features ----
batch["day_of_week"] = batch["batch_date"].dt.dayofweek  # 0=Mon
batch["week_of_year"] = batch["batch_date"].dt.isocalendar().week.astype(int)
batch["month"] = batch["batch_date"].dt.month
batch["day_of_year"] = batch["batch_date"].dt.dayofyear
batch["is_weekend"] = batch["day_of_week"].isin([5, 6]).astype(int)
batch["season"] = batch["month"].apply(get_season)

# ---- 2. Booking lead time - only computable for batches with an actual Tier 1
#      booking timestamp behind them (either Tier 1-only batches, or Group C
#      batches that had a matching Tier 1 booking on the same date) ----
tier1 = pd.read_csv(TIER1_PATH, parse_dates=["booking_date", "start_date"])
tier1["lead_days"] = (tier1["start_date"] - tier1["booking_date"]).dt.days
tier1["batch_date"] = tier1["start_date"].dt.date

lead_time_by_date = tier1.groupby("batch_date")["lead_days"].agg(
    avg_lead_time="mean", median_lead_time="median", n_bookings_for_lead_time="count"
).reset_index()
lead_time_by_date["batch_date"] = pd.to_datetime(lead_time_by_date["batch_date"])

batch = batch.merge(lead_time_by_date, on="batch_date", how="left")
# batches with no Tier 1 booking at all (25 of the 81 Group C batches) get NaN here -
# that's correct and expected, not a bug: there's no booking timestamp to measure from.

# ---- 3. Lag features - computed across the FULL chronological sequence of all
#      110 batches, including inferred-date ones. Confidence only matters for the
#      train/test split (Step 9), not for feature computation. ----
for target in ["participant_count", "class_revenue"]:
    batch[f"{target}_lag_1"] = batch[target].shift(1)
    batch[f"{target}_lag_2"] = batch[target].shift(2)
    batch[f"{target}_lag_4"] = batch[target].shift(4)
    batch[f"{target}_rolling_mean_4"] = batch[target].shift(1).rolling(window=4, min_periods=1).mean()

batch.to_csv(OUT_PATH, index=False)

print(f"Rows written: {len(batch)}")
print(f"Columns: {list(batch.columns)}")
print(f"\nBatches with a computable booking lead time: {batch['avg_lead_time'].notna().sum()} of {len(batch)}")
print(f"Batches with NO Tier 1 booking to measure lead time from: {batch['avg_lead_time'].isna().sum()}")
print(f"\nFirst 4 rows have partial/NaN lag features (expected - not enough prior history yet):")
print(batch[["batch_date", "participant_count", "participant_count_lag_1", "participant_count_rolling_mean_4"]].head(6).to_string(index=False))
print(f"\nSaved -> {OUT_PATH}")