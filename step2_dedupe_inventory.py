import pandas as pd
import re

# ---- Load the Step 1 inventory ----
INVENTORY_PATH = "outputs/sheet_inventory.csv"
OUT_PATH = "outputs/sheet_inventory_deduped.csv"

inv = pd.read_csv(INVENTORY_PATH)

# ---- Client's explicit decisions (do not auto-detect, hard-code them) ----
# "Copy of Class 166" is correct -> drop "Class 166"
# "Class 153" is correct        -> drop "Copy of Class 153"
# "Class 203" and "Class 203B" both dropped entirely (client's call, 2026-09)
DROP_SHEETS = ["Class 166", "Copy of Class 153", "Class 203", "Class 203B"]

before = len(inv)
dropped = inv[inv["sheet_name"].isin(DROP_SHEETS)]
inv = inv[~inv["sheet_name"].isin(DROP_SHEETS)].reset_index(drop=True)
after = len(inv)

print(f"Sheets before: {before}, after: {after}, dropped: {before - after}")
print(dropped[["sheet_name", "row_count"]].to_string(index=False))

# ---- Sanity check: any OTHER near-duplicate sheet names we haven't confirmed? ----
# Flags sheets that share the same sequence number in their name (e.g. two
# sheets both parsing to #166) that are NOT the pair we already resolved.
number_pat = re.compile(r'(class|batch|cf)', re.IGNORECASE)
digits_pat = re.compile(r'\d+')


def extract_num(name):
    if number_pat.search(name):
        digits = digits_pat.findall(name)
        if digits:
            return int(digits[-1])
    return None


inv["_seq"] = inv["sheet_name"].apply(extract_num)
dupe_groups = inv[inv["_seq"].notna()].groupby("_seq")["sheet_name"].apply(list)
remaining_dupes = dupe_groups[dupe_groups.apply(len) > 1]

if len(remaining_dupes) > 0:
    print("\n[WARNING] Other sheets still sharing the same sequence number "
          "(not covered by the client's decision - flag before Step 3):")
    print(remaining_dupes.to_string())
else:
    print("\nNo other duplicate sequence numbers found.")

inv = inv.drop(columns=["_seq"])
inv.to_csv(OUT_PATH, index=False)
print(f"\nSaved de-duplicated inventory -> {OUT_PATH}")
