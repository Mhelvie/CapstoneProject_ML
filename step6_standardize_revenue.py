import re
import pandas as pd

PATH = "data/raw/Copy of Copy of Camp Booking Registration (Responses).xlsx"
PARTICIPANTS_PATH = "outputs/participant_counts.csv"
OUT_PATH = "outputs/revenue_standardized_v2.csv"

DROPPED = {"Class 166", "Copy of Class 153", "Class 203", "Class 203B"}
NON_BOOKING = {"2024", "Form Responses 1", "List", "Sheet3", "Sheet4"}

PRICE = {
    "Discovery": 4250,
    "Refinement": 4100,
    "Non-certified freedivers": 3300,
    "Fundive (certified freedivers)": 2500,
}
PRICE_TO_TYPE = {v: k for k, v in PRICE.items()}

# Direct, unambiguous class-type labels (client-confirmed).
CLASS_TYPE_MAP = {
    "intro dive": "Discovery",
    "practice dive": "Refinement",
    "fundive w/ asst": "Non-certified freedivers",
    "fundive": "Fundive (certified freedivers)",
    "fun dive": "Fundive (certified freedivers)",
    "fd": "Fundive (certified freedivers)",
}

# Tokens that appear INSIDE a combo code, e.g. "3 intro 1 fd" -> 3x intro, 1x fd.
# Only used inside combo parsing, not as standalone labels (a standalone "intro"
# is an occasion/package label per the client, not automatically Discovery).
COMBO_TOKEN_MAP = {
    "intro": "Discovery",
    "fd": "Fundive (certified freedivers)",
}

# Excluded entirely - staffing/admin/accommodation entries, or services outside
# this system's scope (line training = freedive certification, not covered here).
EXCLUDED_LABELS = {
    "coach", "food", "tent", "private room", "staycation", "stayprivate",
    "privatepackage", "lexlunch", "joshmeals", "line training",
}

# Combo tokens that mean "excluded entirely" rather than "unpriced occasion" -
# e.g. "4 lt" = 4 people doing line training, out of scope like standalone "line training".
EXCLUDED_COMBO_TOKENS = {"lt"}

# Occasion/celebration/package labels: count the participant (for staffing),
# but keep their revenue OUT of the class-based forecasting target. Their
# actual recorded payment is reported separately as package revenue.
OCCASION_LABELS = {
    "intro", "private id", "private fd", "private sc", "dtfd", "daytourfd",
    "daytourid", "half dt", "day tour", "wave", "eme",
    "200celeb", "introalacarte", "introprivatealacarte", "pdprivatealacarte",
    "stayalacarte",
}

# Event-echo labels: the Class column just repeats the special-event name.
# Most are fundivers per the client, but check actual payment first.
EVENT_ECHO_LABELS = {"yep", "ilocos", "ilocos2", "halloween dive", "year end dive"}

COMBO_PAT = re.compile(r"(\d+)\s*([a-z]+)", re.IGNORECASE)


def normalize_class_label(value):
    if pd.isna(value):
        return None
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def get_amount(row, *col_names):
    for c in col_names:
        if c in row.index:
            val = pd.to_numeric(row[c], errors="coerce")
            if pd.notna(val):
                return val
    return None


def classify_participant(label, payment_amount):
    """Returns a list of (canonical_type_or_None, price_or_None, revenue_bucket) -
    a list because combo rows expand into more than one participant."""
    if label is None:
        return [(None, payment_amount, "package/occasion revenue (blank class label, using recorded payment)")]

    if label in EXCLUDED_LABELS:
        return []  # not a participant at all

    if label in CLASS_TYPE_MAP:
        ctype = CLASS_TYPE_MAP[label]
        return [(ctype, PRICE[ctype], "class revenue (direct match)")]

    if label in EVENT_ECHO_LABELS:
        # Prefer the participant's own recorded payment if it matches a known price.
        if payment_amount in PRICE_TO_TYPE:
            ctype = PRICE_TO_TYPE[payment_amount]
            return [(ctype, PRICE[ctype], "class revenue (event-echo, payment-matched)")]
        ctype = "Fundive (certified freedivers)"
        return [(ctype, PRICE[ctype], "class revenue (event-echo, defaulted to fundive)")]

    if label in OCCASION_LABELS:
        return [(None, payment_amount, "package/occasion revenue (excluded from class forecast)")]

    if COMBO_PAT.match(label):
        results = []
        for count_str, token in COMBO_PAT.findall(label):
            count = int(count_str)
            token_lower = token.lower()
            if token_lower in EXCLUDED_COMBO_TOKENS:
                continue  # e.g. "lt" (line training) - out of scope, not counted at all
            ctype = COMBO_TOKEN_MAP.get(token_lower)
            if ctype:
                results.extend([(ctype, PRICE[ctype], "class revenue (combo, expanded)")] * count)
            else:
                # unrecognized combo token -> occasion bucket, no fixed price known yet,
                # split any recorded payment evenly across the group
                share = payment_amount / count if payment_amount else None
                results.extend([(None, share, f"package/occasion revenue (combo token '{token}' unpriced)")] * count)
        return results

    return [(None, payment_amount, "package/occasion revenue (unrecognized label, using recorded payment)")]


participants = pd.read_csv(PARTICIPANTS_PATH, keep_default_na=False)
participants["source_row"] = pd.to_numeric(participants["source_row"], errors="coerce")
participants["tier2_only"] = participants["tier2_only"].map({"True": True, "False": False}).fillna(False)

workbook = pd.ExcelFile(PATH)
sheet_rows = []

# ---- Groups A/B unchanged: revenue = recorded down payment, 1:1 by source_row ----
for sheet, group_label in [("2024", "A"), ("Form Responses 1", "B")]:
    raw = pd.read_excel(PATH, sheet_name=sheet, header=0)
    for source_row, row in raw.iterrows():
        sheet_rows.append({
            "source_group": group_label, "source_sheet": sheet, "source_row": source_row,
            "corrected_participant_count": None,  # Tier 1 count is unaffected, use Step 5's value
            "class_revenue": row.get("Amount of down payment sent"),
            "package_revenue": 0,
            "revenue_basis": "down payment (Tier 1)",
        })

# ---- Group C: classify + expand every participant, then sum per sheet ----
for sheet in workbook.sheet_names:
    if sheet in NON_BOOKING or sheet in DROPPED:
        continue
    raw = pd.read_excel(PATH, sheet_name=sheet, header=0)
    if "Name" not in raw.columns:
        continue
    people = raw[raw["Name"].notna()]

    corrected_count = 0
    class_revenue = 0.0
    package_revenue = 0.0
    bucket_counts = {}

    for _, row in people.iterrows():
        label = normalize_class_label(row.get("Class"))
        payment = get_amount(row, "Down Payment", "Fee")
        for ctype, amount, bucket in classify_participant(label, payment):
            corrected_count += 1
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if bucket.startswith("class revenue"):
                class_revenue += amount or 0
            elif bucket.startswith("package/occasion"):
                package_revenue += amount or 0
            # "unmapped - ..." buckets contribute to neither total on purpose - visible via bucket_counts

    sheet_rows.append({
        "source_group": "C", "source_sheet": sheet, "source_row": None,
        "corrected_participant_count": corrected_count,
        "class_revenue": class_revenue,
        "package_revenue": package_revenue,
        "revenue_basis": "; ".join(f"{k}: {v}" for k, v in sorted(bucket_counts.items())),
    })

rev_df = pd.DataFrame(sheet_rows)
out = participants.merge(rev_df, on=["source_group", "source_sheet", "source_row"], how="left")

# Use the corrected count where we have one (Group C); Tier 1 keeps Step 5's count.
out["participant_count"] = out["corrected_participant_count"].fillna(out["participant_count"])
out = out.drop(columns=["corrected_participant_count"])

out.loc[out["tier2_only"] == True, ["class_revenue", "package_revenue"]] = [pd.NA, pd.NA]
out.to_csv(OUT_PATH, index=False)

group_c_out = out[out["source_group"] == "C"]
print(f"Rows written: {len(out)}")
print(f"Group C corrected total participants: {int(group_c_out['participant_count'].sum())} "
      f"(was 2,457 before combo-expansion)")
print(f"Group C total class revenue (forecast target): {group_c_out['class_revenue'].sum():,.0f} PHP")
print(f"Group C total package/occasion revenue (separate): {group_c_out['package_revenue'].sum():,.0f} PHP")
print(f"Saved -> {OUT_PATH}")