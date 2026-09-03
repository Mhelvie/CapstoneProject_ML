import pandas as pd
import re

PATH = "data/raw/Copy of Copy of Camp Booking Registration (Responses).xlsx"

xls = pd.ExcelFile(PATH)
all_sheets = xls.sheet_names

# ---- 1. Split off the non-booking reference sheets (Group D) ----
GROUP_D = {"List", "Sheet3", "Sheet4"}

# ---- 2. Name the two Google Form sheets explicitly (Groups A & B) ----
GROUP_A = ["2024"]
GROUP_B = ["Form Responses 1"]

# ---- 3. Everything else is Group C ----
group_c = [s for s in all_sheets if s not in GROUP_D | set(GROUP_A) | set(GROUP_B)]

# ---- 4. Within Group C, split numbered vs named/special ----
# take the LAST number in the sheet name (handles "CF26-199" -> 199, not 26)
number_pat = re.compile(r'(class|batch|cf)', re.IGNORECASE)
digits_pat = re.compile(r'\d+')

numbered, named = [], []
for s in group_c:
    if number_pat.search(s):
        digits = digits_pat.findall(s)
        if digits:
            numbered.append((s, int(digits[-1])))
            continue
    named.append(s)

numbered.sort(key=lambda x: x[1])

# ---- 5. Build one inventory table with row counts per sheet ----
rows = []
def count_rows(sheet, group):
    df = pd.read_excel(PATH, sheet_name=sheet, header=0)
    if group.startswith("C") or group.startswith("D"):
        # Group C/D sheets: a template row's "Number" column can be
        # pre-filled even with no participant in it. Count rows with
        # an actual Name value instead of just "not fully blank".
        name_col = next((c for c in df.columns if str(c).strip().lower() == "name"), None)
        if name_col is not None:
            return int(df[name_col].notna().sum())
        return len(df.dropna(how="all"))
    return len(df.dropna(how="all"))

for s in GROUP_A:
    rows.append(("A", s, count_rows(s, "A"), "Confirmed date (Timestamp + Date of Class)"))
for s in GROUP_B:
    rows.append(("B", s, count_rows(s, "B"), "Confirmed date, year inferred from Timestamp"))
for s, n in numbered:
    rows.append(("C - numbered", s, count_rows(s, "C"), f"seq #{n}, no date field"))
for s in named:
    rows.append(("C - named/special", s, count_rows(s, "C"), "no number, no date field"))
for s in sorted(GROUP_D):
    rows.append(("D - reference", s, count_rows(s, "D"), "not a booking record"))

inventory = pd.DataFrame(rows, columns=["group", "sheet_name", "row_count", "notes"])

out_path = "outputs/sheet_inventory.csv"
inventory.to_csv(out_path, index=False)

print(f"Total sheets: {len(all_sheets)}")
print(inventory["group"].value_counts())
print()
print(inventory.head(15).to_string(index=False))
print(f"\nSaved full inventory -> {out_path}")