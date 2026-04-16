import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv('data/raw/Asm1_dataset26.csv')

# ── 1. BASIC INFO ────────────────────────────────
print("=" * 60)
print("📊 BASIC INFO")
print("=" * 60)
print(df.info())

# ── 2. SHAPE ─────────────────────────────────────
print("\n" + "=" * 60)
print("📐 SHAPE")
print("=" * 60)
print(f"Total Rows    : {df.shape[0]}")
print(f"Total Columns : {df.shape[1]}")

# ── 3. COLUMN NAMES & DATA TYPES ─────────────────
print("\n" + "=" * 60)
print("📌 COLUMN NAMES & DATA TYPES")
print("=" * 60)
for col in df.columns:
    print(f"  {col:<30} → {df[col].dtype}")

# ── 4. FIRST 5 ROWS ───────────────────────────────
print("\n" + "=" * 60)
print("👀 FIRST 5 ROWS")
print("=" * 60)
print(df.head())

# ── 5. LAST 5 ROWS ────────────────────────────────
print("\n" + "=" * 60)
print("👀 LAST 5 ROWS")
print("=" * 60)
print(df.tail())

# ── 6. MISSING VALUES ─────────────────────────────
print("\n" + "=" * 60)
print("❌ MISSING VALUES PER COLUMN")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count' : missing,
    'Missing %'     : missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0]
      .sort_values('Missing Count', ascending=False)
      .to_string())
if missing.sum() == 0:
    print("  ✅ No missing values found!")

# ── 7. DUPLICATE ROWS ─────────────────────────────
print("\n" + "=" * 60)
print("🔁 DUPLICATE ROWS")
print("=" * 60)
dupes = df.duplicated().sum()
print(f"  Total duplicate rows: {dupes}")

# ── 8. UNIQUE VALUES PER COLUMN ───────────────────
print("\n" + "=" * 60)
print("🔢 UNIQUE VALUES PER COLUMN")
print("=" * 60)
for col in df.columns:
    unique_count = df[col].nunique()
    sample = df[col].dropna().unique()[:3]
    print(f"  {col:<30} → {unique_count} unique | sample: {list(sample)}")

# ── 9. STATISTICS (numeric columns) ───────────────
print("\n" + "=" * 60)
print("📈 STATISTICS (Numeric Columns)")
print("=" * 60)
print(df.describe().to_string())

# ── 10. STATISTICS (text columns) ─────────────────
print("\n" + "=" * 60)
print("🔤 STATISTICS (Text Columns)")
print("=" * 60)
text_cols = df.select_dtypes(include='object')
if not text_cols.empty:
    print(text_cols.describe().to_string())
else:
    print("  No text columns found")



# ── 12. VALUE COUNTS (first 5 cols only) ──────────
print("\n" + "=" * 60)
print("📊 TOP 5 VALUES PER COLUMN")
print("=" * 60)
for col in df.columns:
    print(f"\n  📌 {col}:")
    print(df[col].value_counts().head(5).to_string())

print("\n" + "=" * 60)
print("✅ DATA STRUCTURE REPORT COMPLETE")
print("=" * 60)