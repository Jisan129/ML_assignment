import pandas as pd
import re

# Load your CSV
df = pd.read_csv('data/raw/Asm1_dataset26.csv')

# ── FUNCTION ─────────────────────────────────────
def has_unexpected_chars(val):

    # ❌ flag null/NaN
    if pd.isna(val):
        return True

    val_str = str(val).strip()

    # ❌ flag empty string or only spaces
    if val_str == '':
        return True

    # ✅ allow letters, numbers, dot, space between words
    # but the value must START and END with a letter or number
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9.\s]*[a-zA-Z0-9.]$|^[a-zA-Z0-9.]$'
    return not bool(re.match(pattern, val_str))

# ── COUNT BAD VALUES PER COLUMN ──────────────────
print("=" * 60)
print("📊 BAD VALUES COUNT PER COLUMN")
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