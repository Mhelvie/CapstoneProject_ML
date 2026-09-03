import pandas as pd
import re
import openpyxl
from collections import Counter
from datetime import timedelta

PATH = "data/raw/Copy of Copy of Camp Booking Registration (Responses).xlsx"
TIER1_PATH = "outputs/tier1_cleaned.csv"
OUT_PATH = "outputs/group_c_dates.csv"

MIN_MATCHES = 3
MIN_AGREEMENT = 0.5
DEFAULT_CADENCE_DAYS = 7  # fallback spacing where no anchor exists at all nearby

DROPPED = {"Class 166", "Copy of Class 153", "Class 203", "Class 203B"}
NON_BOOKING = {"2024", "Form Responses 1", "List", "Sheet3", "Sheet4"}
UNANCHORABLE = {"YEP", "isha bday"}


def norm_name(name):
    if pd.isna(name):
        return None
    name = re.sub(r"\(.*?\)", "", str(name))  # strip '(Bea)' style companion notes
    return re.sub(r"\s+", " ", name).strip().lower()


# ---- 1. Build the Tier-1 name -> [dates] lookup from the dated Google Form sheets ----
tier1 = pd.read_csv(TIER1_PATH, parse_dates=["start_date"])
companion_pat = re.compile(r"^\d+\.\s*Full name")
name_to_dates = {}

for sheet in ["2024", "Form Responses 1"]:
    raw = pd.read_excel(PATH, sheet_name=sheet, header=0)
    name_cols = [c for c in raw.columns if companion_pat.match(c)]
    sub = tier1[tier1["source_sheet"] == sheet]
    for idx, row in raw.iterrows():
        match = sub[sub["source_row"] == idx]
        if match.empty:
            continue
        date = match.iloc[0]["start_date"].date()
        for c in name_cols:
            nm = norm_name(row[c])
            if nm:
                name_to_dates.setdefault(nm, []).append(date)

# ---- 2. Walk the workbook's own tab order (confirmed to run newest -> oldest) ----
wb = openpyxl.load_workbook(PATH, data_only=True)
group_c_ordered = [s for s in wb.sheetnames if s not in NON_BOOKING and s not in DROPPED]
group_c_oldest_first = list(reversed(group_c_ordered))  # walk oldest -> newest

# ---- 3. Compute a name-matched candidate anchor for every Group C sheet ----
candidates = []
for sheet in group_c_oldest_first:
    df = pd.read_excel(PATH, sheet_name=sheet, header=0)
    if "Name" not in df.columns:
        candidates.append({"sheet": sheet, "n_participants": len(df), "n_matches": 0,
                           "mode_date": None, "agreement": 0.0})
        continue
    names = df["Name"].dropna().apply(norm_name)
    matched_dates = []
    for n in names:
        matched_dates.extend(name_to_dates.get(n, []))
    if matched_dates:
        mode_date, mode_count = Counter(matched_dates).most_common(1)[0]
        agreement = mode_count / len(matched_dates)
    else:
        mode_date, agreement = None, 0.0
    candidates.append({"sheet": sheet, "n_participants": len(names), "n_matches": len(matched_dates),
                       "mode_date": mode_date, "agreement": round(agreement, 2)})

cand_df = pd.DataFrame(candidates).set_index("sheet").loc[group_c_oldest_first]

# ---- 4. Accept anchors that clear the confidence bar and move forward in time ----
accepted = {}
rejected = []
last_accepted_date = None
for sheet, row in cand_df.iterrows():
    ok_confidence = row["n_matches"] >= MIN_MATCHES and row["agreement"] >= MIN_AGREEMENT
    if ok_confidence and row["mode_date"] is not None:
        if last_accepted_date is None or row["mode_date"] > last_accepted_date:
            accepted[sheet] = row["mode_date"]
            last_accepted_date = row["mode_date"]
            continue
        rejected.append((sheet, row["mode_date"], "would go backwards vs. previous anchor"))

# ---- 5. Fill remaining sheets by interpolation or weekly edge extrapolation ----
positions = {s: i for i, s in enumerate(group_c_oldest_first)}
anchor_positions = sorted(positions[s] for s in accepted)

final_dates = {}
for sheet in group_c_oldest_first:
    if sheet in UNANCHORABLE:
        final_dates[sheet] = (None, "tier 2 only - no date clue")
        continue
    if sheet in accepted:
        final_dates[sheet] = (accepted[sheet], "name-matched anchor")
        continue
    pos = positions[sheet]
    before = [p for p in anchor_positions if p < pos]
    after = [p for p in anchor_positions if p > pos]
    if before and after:
        p0, p1 = before[-1], after[0]
        s0, s1 = group_c_oldest_first[p0], group_c_oldest_first[p1]
        d0, d1 = accepted[s0], accepted[s1]
        frac = (pos - p0) / (p1 - p0)
        interp_date = d0 + timedelta(days=round((d1 - d0).days * frac))
        final_dates[sheet] = (interp_date, "interpolated (between anchors)")
    elif before:
        p0 = before[-1]
        steps = pos - p0
        final_dates[sheet] = (accepted[group_c_oldest_first[p0]] + timedelta(days=DEFAULT_CADENCE_DAYS * steps),
                              "extrapolated (weekly cadence, no later anchor)")
    elif after:
        p1 = after[0]
        steps = p1 - pos
        final_dates[sheet] = (accepted[group_c_oldest_first[p1]] - timedelta(days=DEFAULT_CADENCE_DAYS * steps),
                              "extrapolated (weekly cadence, no earlier anchor)")
    else:
        final_dates[sheet] = (None, "no anchor available at all - needs client input")

# ---- 6. Assemble and save ----
out_rows = []
for sheet in group_c_ordered:  # save newest-first to match the workbook's own order
    date, source = final_dates[sheet]
    row = cand_df.loc[sheet]
    out_rows.append({
        "sheet_name": sheet,
        "assigned_date": date,
        "date_confidence": "inferred",
        "date_source": source,
        "n_participants": row["n_participants"],
        "candidate_from_matching": row["mode_date"],
        "match_agreement": row["agreement"],
        "n_name_matches": row["n_matches"],
    })

out_df = pd.DataFrame(out_rows)
out_df.to_csv(OUT_PATH, index=False)

n_anchored = sum(1 for s in group_c_ordered if final_dates[s][1] == "name-matched anchor")
n_interp = sum(1 for s in group_c_ordered if "interpolated" in final_dates[s][1])
n_extrap = sum(1 for s in group_c_ordered if "extrapolated" in final_dates[s][1])
n_none = sum(1 for s in group_c_ordered if final_dates[s][0] is None)

print(f"Group C sheets processed: {len(group_c_ordered)}")
print(f"  name-matched anchors accepted : {n_anchored}")
print(f"  interpolated between anchors  : {n_interp}")
print(f"  extrapolated (edge, no anchor) : {n_extrap}")
print(f"  no date at all                : {n_none}")

if rejected:
    print(f"\nRejected {len(rejected)} candidate anchor(s) as inconsistent with sequence order:")
    for sheet, date, reason in rejected:
        print(f"   {sheet}: candidate {date} - {reason}")

if n_none:
    print("\nSheets with NO usable anchor on either side (need direct client input):")
    for s in group_c_ordered:
        if final_dates[s][0] is None:
            print(f"   {s}")

print(f"\nSaved -> {OUT_PATH}")
