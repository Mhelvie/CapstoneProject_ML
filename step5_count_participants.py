import pandas as pd
import re

PATH = "data/raw/Copy of Copy of Camp Booking Registration (Responses).xlsx"
TIER1_PATH = "outputs/tier1_cleaned.csv"
GROUP_C_DATES_PATH = "outputs/group_c_dates.csv"
OUT_PATH = "outputs/participant_counts.csv"

DROPPED = {"Class 166", "Copy of Class 153", "Class 203", "Class 203B"}
NON_BOOKING = {"2024", "Form Responses 1", "List", "Sheet3", "Sheet4"}
COMPANION_PAT = re.compile(r"^\d+\.\s*Full name", re.IGNORECASE)


def non_empty_count(row, columns):
    return int(row[columns].notna().sum())


# ---- Groups A/B: count the primary booker plus non-empty companions ----
tier1 = pd.read_csv(TIER1_PATH, parse_dates=["booking_date", "start_date", "end_date"])
form_rows = []
for sheet in ["2024", "Form Responses 1"]:
    raw = pd.read_excel(PATH, sheet_name=sheet, header=0)
    companion_cols = [
        c for c in raw.columns
        if COMPANION_PAT.match(str(c)) and "Primary" not in str(c)
    ]
    dated_rows = tier1[tier1["source_sheet"] == sheet]
    for _, dated_row in dated_rows.iterrows():
        source_row = int(dated_row["source_row"])
        raw_row = raw.iloc[source_row]
        participant_count = 1 + non_empty_count(raw_row, companion_cols)
        form_rows.append({
            "source_group": dated_row["source_group"],
            "source_sheet": sheet,
            "source_row": source_row,
            "participant_count": participant_count,
            "assigned_date": dated_row["start_date"].date(),
            "date_confidence": dated_row["date_confidence"],
            "date_source": "confirmed Tier 1",
            "tier2_only": False,
        })

# ---- Group C: each non-empty participant Name row counts as one participant ----
group_c_dates = pd.read_csv(GROUP_C_DATES_PATH, parse_dates=["assigned_date"])
wb_sheets = pd.ExcelFile(PATH).sheet_names
group_c_sheets = [
    sheet for sheet in wb_sheets
    if sheet not in NON_BOOKING and sheet not in DROPPED
]

group_c_rows = []
for sheet in group_c_sheets:
    raw = pd.read_excel(PATH, sheet_name=sheet, header=0)
    if "Name" in raw.columns:
        participant_count = int(raw["Name"].notna().sum())
    else:
        participant_count = int(len(raw.dropna(how="all")))

    date_row = group_c_dates[group_c_dates["sheet_name"] == sheet]
    if date_row.empty:
        assigned_date = None
        date_confidence = "inferred"
        date_source = "missing Step 4 result"
    else:
        date_value = date_row.iloc[0]["assigned_date"]
        assigned_date = date_value.date() if pd.notna(date_value) else None
        date_confidence = date_row.iloc[0]["date_confidence"]
        date_source = date_row.iloc[0]["date_source"]

    # NOTE: the old TIER2_ONLY={"YEP","isha bday"} override that forced
    # assigned_date=None here has been removed - Step 4's fixed number-based
    # ordering now gives both of those sheets real (if lower-confidence)
    # dates, and this script should just pass those through.

    group_c_rows.append({
        "source_group": "C",
        "source_sheet": sheet,
        "source_row": None,
        "participant_count": participant_count,
        "assigned_date": assigned_date,
        "date_confidence": date_confidence,
        "date_source": date_source,
        "tier2_only": assigned_date is None,
    })

out_df = pd.DataFrame(form_rows + group_c_rows)
out_df.to_csv(OUT_PATH, index=False)

print(f"Rows written: {len(out_df)}")
print(f"Total participants represented: {out_df['participant_count'].sum()}")
print(f"Tier 2-only sheets: {int(out_df['tier2_only'].sum())}")
print(f"Saved -> {OUT_PATH}")