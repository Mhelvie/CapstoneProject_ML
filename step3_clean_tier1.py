import pandas as pd
import re
import dateutil.parser as dp

PATH = "data/raw/Copy of Copy of Camp Booking Registration (Responses).xlsx"
OUT_PATH = "outputs/tier1_cleaned.csv"

COMPANION_COLS = [f"{i}. Full name  (Companion)" for i in range(2, 16)]
# note: column 2's header has a single space ("2. Full name (Companion)"),
# columns 3-15 have a double space ("3. Full name  (Companion)") - handled below

SPLIT_PAT = re.compile(r"\s+to\s+|\s*-\s*", re.IGNORECASE)
MONTH_PAT = re.compile(r"^[A-Za-z]+")


def parse_one_range(text, year):
    """Parse a single 'Month D to D' / 'Month D-D' / 'Month D to Month D' chunk."""
    parts = SPLIT_PAT.split(text.strip())
    if len(parts) != 2:
        return None  # unrecognized shape - caller will flag for review
    start_txt, end_txt = parts[0].strip(), parts[1].strip()
    month_match = MONTH_PAT.match(start_txt)
    if not month_match:
        return None
    month_name = month_match.group(0)

    try:
        start_date = dp.parse(f"{start_txt} {year}").date()
    except (ValueError, OverflowError):
        return None

    # end fragment may be "18" (needs the month prepended) or "June 01" (already has one)
    if MONTH_PAT.match(end_txt):
        end_source = f"{end_txt} {year}"
    else:
        end_source = f"{month_name} {end_txt} {year}"
    try:
        end_date = dp.parse(end_source).date()
    except (ValueError, OverflowError):
        return None

    if end_date < start_date:
        # cross-month/cross-year roll-over dateutil didn't catch on its own
        end_date = end_date.replace(year=end_date.year + 1)

    return start_date, end_date


def resolve_year(class_month_text, booking_timestamp):
    """Year comes from the booking timestamp, with a Dec-booking/Jan-class rollover check."""
    ts_year = booking_timestamp.year
    ts_month = booking_timestamp.month
    class_month_word = MONTH_PAT.match(class_month_text.strip())
    if class_month_word:
        try:
            class_month_num = dp.parse(class_month_word.group(0) + " 1 2000").month
        except ValueError:
            return ts_year
        if class_month_num == 1 and ts_month == 12:
            return ts_year + 1
    return ts_year


def load_group(sheet_name, group_label, path=PATH):
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)
    df = df.drop(columns=[c for c in df.columns if c == "Date of Class.1"])

    # companion-name columns vary slightly in spacing; match by prefix instead
    companion_cols = [c for c in df.columns if re.match(r"^\d+\.\s*Full name", c) and "Primary" not in c]

    records = []
    skipped = []
    for idx, row in df.iterrows():
        raw_text = row.get("Date of Class")
        ts = row.get("Timestamp")
        if pd.isna(raw_text) or pd.isna(ts):
            skipped.append((sheet_name, idx, raw_text, "missing date or timestamp"))
            continue

        participant_count = 1 + row[companion_cols].notna().sum()
        down_payment = row.get("Amount of down payment sent")
        mobile = row.get("Mobile Number")

        # a booking can list more than one weekend, comma-separated
        chunks = [c.strip() for c in str(raw_text).split(",") if c.strip()]
        for chunk in chunks:
            year = resolve_year(chunk, ts)
            parsed = parse_one_range(chunk, year)
            if parsed is None:
                skipped.append((sheet_name, idx, raw_text, f"could not parse chunk: '{chunk}'"))
                continue
            start_date, end_date = parsed
            records.append({
                "source_group": group_label,
                "source_sheet": sheet_name,
                "source_row": idx,
                "booking_date": ts,
                "class_date_raw": raw_text,
                "start_date": start_date,
                "end_date": end_date,
                "multi_batch_booking": len(chunks) > 1,
                "mobile_number": mobile,
                "participant_count": participant_count,
                "down_payment": down_payment,
                "date_confidence": "confirmed",
            })
    return pd.DataFrame(records), skipped


a_df, a_skipped = load_group("2024", "A")
b_df, b_skipped = load_group("Form Responses 1", "B")

tier1 = pd.concat([a_df, b_df], ignore_index=True)
tier1 = tier1.sort_values("start_date").reset_index(drop=True)
tier1.to_csv(OUT_PATH, index=False)

print(f"Group A: {len(a_df)} class-batch rows from {a_df['source_row'].nunique()} bookings "
      f"({len(a_skipped)} skipped)")
print(f"Group B: {len(b_df)} class-batch rows from {b_df['source_row'].nunique()} bookings "
      f"({len(b_skipped)} skipped)")
print(f"Multi-batch bookings found: {tier1['multi_batch_booking'].sum()}")
print(f"Date range: {tier1['start_date'].min()} -> {tier1['start_date'].max()}")

all_skipped = a_skipped + b_skipped
if all_skipped:
    print(f"\n⚠ {len(all_skipped)} rows could not be parsed - review these before Step 4:")
    for sheet, idx, raw, reason in all_skipped:
        print(f"  [{sheet} row {idx}] '{raw}' -> {reason}")
else:
    print("\nAll rows parsed successfully.")

print(f"\nSaved -> {OUT_PATH}")