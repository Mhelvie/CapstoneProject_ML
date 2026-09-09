import pandas as pd

FEATURED_PATH = "outputs/featured_dataset.csv"
OUT_PATH = "outputs/split_dataset.csv"

df = pd.read_csv(FEATURED_PATH, parse_dates=["batch_date"])
df = df.sort_values("batch_date").reset_index(drop=True)

# ---- Exclude the extrapolated tail from the split candidate pool entirely.
#      These are the batches with no real anchor on either side (chained off a
#      default weekly-cadence guess) - they're not reliable enough to train,
#      validate, OR test on. Treat them as future-forecast targets to predict
#      later, not as historical data to learn from. They stay in the CSV with
#      split=None so nothing is silently dropped from the file. ----
is_extrapolated = df["date_source"].str.contains("extrapolated", na=False)
eligible = df[~is_extrapolated].copy()
print(f"Excluding {is_extrapolated.sum()} extrapolated-date batches from the split pool "
      f"(kept in the file, just not used for train/val/test).")

n = len(eligible)
n_train_target = int(round(n * 0.70))
n_val_target = int(round(n * 0.15))
n_test_target = n - n_train_target - n_val_target

# ---- Build the test set by walking backward from the most recent batch,
#      skipping any batch whose date isn't 'confirmed'. This can leave gaps
#      (a confirmed batch surrounded by inferred ones on both sides) - that's
#      expected, since date_confidence for EVERY Group C batch is 'inferred'
#      even when it came from a solid name-matched anchor - only Tier 1-only
#      batches (advance bookings with no roster) are ever 'confirmed' here. ----
# ---- Build the test set from the tail of the CONFIRMED batches only. Then -
#      critically - exclude every batch (confirmed or inferred) whose date falls
#      at or after the test window's start from train/validation. Skipping this
#      step would leak "future" batches into training: an inferred batch dated
#      after the test window's start describes something chronologically later
#      than what the model is being evaluated against, even though its own date
#      is uncertain. Those excluded rows are simply not usable in this
#      evaluation scheme - they're the batches closest to "now", which is a
#      forecasting target for later, not a training input. ----
confirmed_sorted = eligible[
    (eligible["date_confidence"] == "confirmed") | (eligible["date_source"] == "name-matched anchor")
].sort_values("batch_date")
# Name-matched anchors are treated as test-eligible alongside truly 'confirmed'
# dates: they were independently validated against real Tier 1 booking dates via
# majority-vote name matching (Step 4), which is strong enough evidence to trust
# for evaluation - unlike interpolated/extrapolated dates, which are estimates
# with no direct confirmation. Without this widening, 'confirmed' dates alone are
# so sparse and scattered across the timeline that the last N of them span almost
# the entire dataset, forcing near-total exclusion of everything else (verified:
# 29 confirmed-only dates span a full year in their final 17; widening to include
# anchors narrows that same window to about 6 months).
if len(confirmed_sorted) < n_test_target:
    print(f"⚠ Only {len(confirmed_sorted)} confirmed batches exist - wanted {n_test_target} for the test set. "
          f"Using all of them; test set will be smaller than 15%.")
    n_test_target = len(confirmed_sorted)

test_rows = confirmed_sorted.tail(n_test_target)
test_idx = set(test_rows.index)
test_start_date = test_rows["batch_date"].min()

excluded_idx = set(eligible[(eligible["batch_date"] >= test_start_date) & (~eligible.index.isin(test_idx))].index)
remaining_idx = [i for i in eligible.index if i not in test_idx and i not in excluded_idx]

n_train = min(n_train_target, len(remaining_idx))
train_idx = set(remaining_idx[:n_train])
val_idx = set(remaining_idx[n_train:])

df["split"] = None
df.loc[df.index.isin(train_idx), "split"] = "train"
df.loc[df.index.isin(val_idx), "split"] = "validation"
df.loc[df.index.isin(test_idx), "split"] = "test"
# rows in excluded_idx, and the extrapolated rows excluded up front, keep split = None

df.to_csv(OUT_PATH, index=False)

print(f"Total batches: {n}")
print(f"  train:      {(df['split'] == 'train').sum()}  "
      f"({df.loc[df['split']=='train','batch_date'].min().date()} -> {df.loc[df['split']=='train','batch_date'].max().date()})")
print(f"  validation: {(df['split'] == 'validation').sum()}  "
      f"({df.loc[df['split']=='validation','batch_date'].min().date()} -> {df.loc[df['split']=='validation','batch_date'].max().date()})")
print(f"  test:       {(df['split'] == 'test').sum()}  "
      f"({df.loc[df['split']=='test','batch_date'].min().date()} -> {df.loc[df['split']=='test','batch_date'].max().date()})")
n_excluded = df["split"].isna().sum()
print(f"  excluded (chronologically at/after test start, not usable in this split): {n_excluded}")

test_confidence = df.loc[df["split"] == "test", ["date_confidence", "date_source"]].drop_duplicates()
print(f"\nTest set date_confidence/date_source combinations present:")
print(test_confidence.to_string(index=False))

train_val_max = df.loc[df["split"].isin(["train", "validation"]), "batch_date"].max()
test_min = df.loc[df["split"] == "test", "batch_date"].min()
print(f"\nLeakage check: latest train/validation date ({train_val_max.date()}) "
      f"{'<=' if train_val_max <= test_min else '>'} earliest test date ({test_min.date()}) "
      f"({'OK - no leakage' if train_val_max <= test_min else '⚠ LEAKAGE DETECTED'})")

print(f"\nSaved -> {OUT_PATH}")