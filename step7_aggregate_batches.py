import pandas as pd
import re

TIER1_PATH = "outputs/tier1_cleaned.csv"
GROUP_C_DATES_PATH = "outputs/group_c_dates.csv"
REVENUE_PATH = "outputs/revenue_standardized_v2.csv"
OUT_PATH = "outputs/batch_level_dataset.csv"

tier1 = pd.read_csv(TIER1_PATH, parse_dates=["start_date"])
group_c_dates = pd.read_csv(GROUP_C_DATES_PATH, parse_dates=["assigned_date"])
revenue = pd.read_csv(REVENUE_PATH, keep_default_na=False)
revenue["source_row"] = pd.to_numeric(revenue["source_row"], errors="coerce")
revenue["tier2_only"] = revenue["tier2_only"].map({"True": True, "False": False}).fillna(False)
revenue["class_revenue"] = pd.to_numeric(revenue["class_revenue"], errors="coerce")
revenue["package_revenue"] = pd.to_numeric(revenue["package_revenue"], errors="coerce")
revenue["participant_count"] = pd.to_numeric(revenue["participant_count"], errors="coerce")

# ---- Tier 1 side: aggregate bookings by their weekend date (multiple households
#      can book the same weekend) ----
tier1["batch_date"] = tier1["start_date"].dt.date
tier1_rev = revenue[revenue["source_group"].isin(["A", "B"])]
tier1_merged = tier1.merge(
    tier1_rev[["source_group", "source_sheet", "source_row", "class_revenue"]],
    on=["source_group", "source_sheet", "source_row"],
    how="left",
)
tier1_by_date = tier1_merged.groupby("batch_date").agg(
    tier1_booking_count=("source_row", "count"),
    tier1_participant_count=("participant_count", "sum"),
    tier1_revenue=("class_revenue", "sum"),
).reset_index()

# ---- Group C side: one row per sheet, already batch-level ----
group_c_rev = revenue[revenue["source_group"] == "C"][
    ["source_sheet", "participant_count", "class_revenue", "package_revenue", "tier2_only"]
]
group_c = group_c_dates.merge(group_c_rev, left_on="sheet_name", right_on="source_sheet", how="left")
group_c["batch_date"] = group_c["assigned_date"].dt.date

# ---- Combine without double-counting: where a Group C sheet exists for a date,
#      it's authoritative (actual roster/settlement). Tier 1 only fills in dates
#      that have no Group C sheet at all. ----
group_c_dates_set = set(group_c["batch_date"].dropna())

batch_rows = []

# batches with a Group C roster (use Group C as ground truth; keep Tier 1's
# booking count for that date as a reference metric only, never summed in)
for _, row in group_c.iterrows():
    if pd.isna(row["batch_date"]):
        continue  # shouldn't happen post-Step 4, but guard anyway
    tier1_ref = tier1_by_date[tier1_by_date["batch_date"] == row["batch_date"]]
    batch_rows.append({
        "batch_id": row["sheet_name"],
        "batch_date": row["batch_date"],
        "primary_source": "Group C roster (authoritative)",
        "date_confidence": row["date_confidence"],
        "date_source": row["date_source"],
        "participant_count": row["participant_count"],
        "class_revenue": row["class_revenue"] if not row["tier2_only"] else None,
        "package_revenue": row["package_revenue"] if not row["tier2_only"] else None,
        "tier1_booking_count_same_date": int(tier1_ref["tier1_booking_count"].iloc[0]) if len(tier1_ref) else 0,
        "tier1_had_matching_booking": len(tier1_ref) > 0,
    })

# Tier 1-only batches: dates with advance bookings but no Group C roster at all
tier1_only = tier1_by_date[~tier1_by_date["batch_date"].isin(group_c_dates_set)]
for _, row in tier1_only.iterrows():
    batch_rows.append({
        "batch_id": f"tier1-{row['batch_date']}",
        "batch_date": row["batch_date"],
        "primary_source": "Tier 1 booking (no roster available yet)",
        "date_confidence": "confirmed",
        "date_source": "Tier 1 Google Form",
        "participant_count": row["tier1_participant_count"],
        "class_revenue": row["tier1_revenue"],
        "package_revenue": 0,
        "tier1_booking_count_same_date": int(row["tier1_booking_count"]),
        "tier1_had_matching_booking": True,
    })

batch_df = pd.DataFrame(batch_rows).sort_values("batch_date").reset_index(drop=True)
batch_df.to_csv(OUT_PATH, index=False)

n_groupc_primary = (batch_df["primary_source"] == "Group C roster (authoritative)").sum()
n_tier1_only = (batch_df["primary_source"] == "Tier 1 booking (no roster available yet)").sum()
n_overlap = batch_df["tier1_had_matching_booking"].sum() - n_tier1_only

print(f"Total batches: {len(batch_df)}")
print(f"  Group C roster batches (authoritative): {n_groupc_primary}")
print(f"    - of which also had a Tier 1 booking on the same date (avoided double-count): {n_overlap}")
print(f"  Tier 1-only batches (no roster exists): {n_tier1_only}")
print(f"\nTotal participants across all batches: {int(batch_df['participant_count'].sum())}")
print(f"Total class revenue: {batch_df['class_revenue'].sum():,.0f} PHP")
print(f"Total package/occasion revenue: {batch_df['package_revenue'].sum():,.0f} PHP")
print(f"Date range: {batch_df['batch_date'].min()} -> {batch_df['batch_date'].max()}")
print(f"\nSaved -> {OUT_PATH}")