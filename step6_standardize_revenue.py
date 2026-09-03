import re

import pandas as pd

PATH = "data/raw/Copy of Copy of Camp Booking Registration (Responses).xlsx"
PARTICIPANTS_PATH = "outputs/participant_counts.csv"
OUT_PATH = "outputs/revenue_standardized.csv"

DROPPED = {"Class 166", "Copy of Class 153", "Class 203", "Class 203B"}
NON_BOOKING = {"2024", "Form Responses 1", "List", "Sheet3", "Sheet4"}
TIER2_ONLY = {"YEP", "isha bday"}

PRICE = {
    "Discovery": 4250,
    "Refinement": 4100,
    "Non-certified freedivers": 3300,
    "Fundive (certified freedivers)": 2500,
}

# These aliases reflect the client's supplied mapping. The fd alias needs client confirmation.
CLASS_TYPE_MAP = {
    "intro dive": "Discovery",
    "practice dive": "Refinement",
    "fundive w/ asst": "Non-certified freedivers",
    "fundive": "Fundive (certified freedivers)",
    "fun dive": "Fundive (certified freedivers)",
    "fd": "Fundive (certified freedivers)",
}


def normalize_class_label(value):
    if pd.isna(value):
        return None
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def canonicalize_class_type(value):
    return CLASS_TYPE_MAP.get(normalize_class_label(value))


participants = pd.read_csv(PARTICIPANTS_PATH, keep_default_na=False)
participants["source_row"] = pd.to_numeric(participants["source_row"], errors="coerce")
participants["tier2_only"] = participants["tier2_only"].map({"True": True, "False": False}).fillna(False)
workbook = pd.ExcelFile(PATH)
raw_rows = []

# Read source values needed for class labels and fallback/reference amounts.
for sheet in workbook.sheet_names:
    if sheet in NON_BOOKING or sheet in DROPPED:
        continue
    raw = pd.read_excel(PATH, sheet_name=sheet, header=0)
    if sheet in {"2024", "Form Responses 1"}:
        for source_row, row in raw.iterrows():
            raw_rows.append({
                "source_group": "A" if sheet == "2024" else "B",
                "source_sheet": sheet,
                "source_row": source_row,
                "raw_class_type": None,
                "down_payment_reference": row.get("Amount of down payment sent"),
                "fallback_amount": row.get("Amount of down payment sent"),
            })
        continue

    if "Name" not in raw.columns:
        continue
    for _, row in raw.iterrows():
        raw_rows.append({
            "source_group": "C",
            "source_sheet": sheet,
            "source_row": None,
            "raw_class_type": row.get("Class"),
            "down_payment_reference": row.get("Down Payment"),
            "fallback_amount": row.get("Down Payment"),
        })

raw_df = pd.DataFrame(raw_rows)
raw_df["source_row"] = pd.to_numeric(raw_df["source_row"], errors="coerce")

# Add source-level class labels and payment references to the Step 5 records.
out = participants.merge(
    raw_df,
    on=["source_group", "source_sheet", "source_row"],
    how="left",
)

# Group C sheets have one output row per sheet, so reduce raw labels/payments to sheet level.
group_c_sheet_values = raw_df[raw_df["source_group"] == "C"].groupby("source_sheet")
for sheet, indexes in group_c_sheet_values.groups.items():
    sheet_rows = raw_df.loc[indexes]
    labels = [x for x in sheet_rows["raw_class_type"].dropna().map(normalize_class_label) if x]
    unique_labels = sorted(set(labels))
    label = unique_labels[0] if len(unique_labels) == 1 else None
    out.loc[
        (out["source_group"] == "C") & (out["source_sheet"] == sheet),
        "raw_class_type",
    ] = label
    out.loc[
        (out["source_group"] == "C") & (out["source_sheet"] == sheet),
        "down_payment_reference",
    ] = pd.to_numeric(sheet_rows["down_payment_reference"], errors="coerce").sum(min_count=1)
    out.loc[
        (out["source_group"] == "C") & (out["source_sheet"] == sheet),
        "fallback_amount",
    ] = pd.to_numeric(sheet_rows["fallback_amount"], errors="coerce").sum(min_count=1)

out["class_type_canonical"] = out["raw_class_type"].apply(canonicalize_class_type)
out["fixed_price_php"] = out["class_type_canonical"].map(PRICE)
out["revenue_basis"] = "fixed price x participant count"
out["revenue"] = out["fixed_price_php"] * out["participant_count"]

unknown = out["class_type_canonical"].isna()
out.loc[unknown, "revenue"] = pd.to_numeric(out.loc[unknown, "fallback_amount"], errors="coerce")
out.loc[unknown, "revenue_basis"] = "down payment fallback - class type unknown"
out.loc[out["tier2_only"] == True, "revenue_basis"] = "tier 2 only - no forecasting revenue"
out.loc[out["tier2_only"] == True, "revenue"] = pd.NA

out["mapping_review_required"] = out["raw_class_type"].notna() & unknown
out = out.drop(columns=["fallback_amount"])
out.to_csv(OUT_PATH, index=False)

print(f"Rows written: {len(out)}")
print(f"Fixed-price revenue rows: {int((out['revenue_basis'] == 'fixed price x participant count').sum())}")
print(f"Down-payment fallback rows: {int((out['revenue_basis'] == 'down payment fallback - class type unknown').sum())}")
print(f"Mapping review rows: {int(out['mapping_review_required'].sum())}")
print(f"Tier 2-only rows: {int((out['tier2_only'] == True).sum())}")
print(f"Saved -> {OUT_PATH}")
