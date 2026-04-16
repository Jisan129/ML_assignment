import pandas as pd
import re

# Load your CSV
df = pd.read_csv('data/raw/Asm1_dataset26.csv')


# ── FUNCTION: Check if value has unexpected characters ──
def has_unexpected_chars(val):
    if pd.isna(val):
        return False  # skip nulls

    val_str = str(val)  # convert to string to check

    # This pattern allows only:
    # a-z, A-Z (letters)
    # 0-9 (numbers)
    # . (decimal point)
    # space
    # - (negative numbers)
    pattern = r'^[a-zA-Z0-9.\s\-]+$'

    # If it does NOT match the pattern → it's unexpected
    return not bool(re.match(pattern, val_str))


# ── SCAN ALL COLUMNS ────────────────────────────────────
print("=" * 60)
print("🔍 SCANNING FOR UNEXPECTED CHARACTERS")
print("=" * 60)

# Apply check to every cell
problem_mask = df.apply(
    lambda row: row.apply(has_unexpected_chars).any(),
    axis=1
)

# Get problem rows
problem_rows = df[problem_mask].copy()


# Show WHICH column has the problem
def get_problem_details(row):
    problems = []
    for col in df.columns:
        val = row[col]
        if has_unexpected_chars(val):
            problems.append(f"{col}: '{val}'")
    return ' | '.join(problems)


problem_rows['⚠️ Problem Detail'] = problem_rows.apply(
    get_problem_details, axis=1
)

# ── RESULTS ─────────────────────────────────────────────
print(f"\n📌 Total rows scanned  : {len(df)}")
print(f"✅ Clean rows          : {len(df) - len(problem_rows)}")
print(f"⚠️  Problem rows        : {len(problem_rows)}")

if len(problem_rows) > 0:
    print(f"\n🔴 Problem rows:")
    # print(problem_rows.to_string())

    # Save to CSV
    problem_rows.to_csv('problem_rows.csv', index=True)
    print(f"\n💾 Saved to: problem_rows.csv")
else:
    print("\n🎉 All data looks clean!")

# ── COLUMN SUMMARY ──────────────────────────────────────
print("\n" + "=" * 60)
print("📊 PROBLEM COUNT PER COLUMN")
print("=" * 60)

for col in df.columns:
    count = df[col].apply(has_unexpected_chars).sum()
    if count > 0:
        # Show sample of bad values
        bad_vals = df[col][df[col].apply(has_unexpected_chars)].unique()[:3]
        print(f"\n  📌 Column : {col}")
        print(f"     Count  : {count} rows")
        print(' '.join(str(x) for x in bad_vals))

print("\n" + "=" * 60)
print("✅ SCAN COMPLETE")
print("=" * 60)
bad_columns = []

for col in df.columns:
    bad_count = df[col].apply(has_unexpected_chars).sum()

    if bad_count > 0:
        bad_columns.append(col)
        bad_vals = df[col][df[col].apply(has_unexpected_chars)].unique()[:3]
        print(f"\n  📌 Column  : {col}")
        print(f"     Bad rows : {bad_count}")
        print(f"     Sample   : {', '.join(str(x) for x in bad_vals)}")

# ── SUMMARY ──────────────────────────────────────
print("\n" + "=" * 60)
print("📌 SUMMARY")
print("=" * 60)
print(f"  Total columns         : {len(df.columns)}")
print(f"  Columns with bad data : {len(bad_columns)}")
print(f"  Clean columns         : {len(df.columns) - len(bad_columns)}")
print(f"\n  ⚠️  Bad columns list : {', '.join(bad_columns)}")
print("=" * 60)