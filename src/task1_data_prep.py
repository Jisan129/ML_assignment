import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW / "Asm1_dataset26.csv")
df.columns = df.columns.str.strip()          # strip leading/trailing whitespace
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Replace '--' in Wind Direction columns with NaN ───────────────────────
wind_cols = [
    "Wind Direction_D1_D30",
    "Wind Direction_D31_D60",
    "Wind Direction_D61_D90",
    "Wind Direction_D91_D120",
]
for col in wind_cols:
    before = (df[col] == "--").sum()
    df[col] = df[col].replace("--", np.nan)
    print(f"  Replaced {before} '--' values in '{col}' with NaN")

# ── 3. Impute missing Min temp columns with column mean ──────────────────────
min_temp_cols = [
    "Min temp_D1_D30",
    "Min temp_D31_D60",
    "Min temp_D61_D90",
    "Min temp_D91_D120",
]
print("\nImputing missing Min temp values with column mean:")
for col in min_temp_cols:
    missing = df[col].isna().sum()
    col_mean = df[col].mean()
    df[col] = df[col].fillna(col_mean)
    print(f"  '{col}': filled {missing} NaN(s) with mean={col_mean:.4f}")

# ── 4. Calculate yield_per_hectare ───────────────────────────────────────────
df["yield_per_hectare"] = df["Paddy yield(in Kg)"] / df["Hectares"]
print(f"\nyield_per_hectare — min: {df['yield_per_hectare'].min():.2f}, "
      f"max: {df['yield_per_hectare'].max():.2f}, "
      f"mean: {df['yield_per_hectare'].mean():.2f}")

# ── 5. Create binary target 'isAboveAvg' ─────────────────────────────────────
mean_yield = df["yield_per_hectare"].mean()
df["isAboveAvg"] = (df["yield_per_hectare"] > mean_yield).astype(int)

# ── 6. Print value counts ────────────────────────────────────────────────────
print("\nisAboveAvg value counts:")
print(df["isAboveAvg"].value_counts().to_string())
print(f"  (1 = above average yield_per_hectare > {mean_yield:.2f} kg/ha)")

# ── 7. Label encode categorical columns ─────────────────────────────────────
label_cols = ["Variety", "Soil Types", "Nursery"]
le = LabelEncoder()
print("\nLabel encoding:")
for col in label_cols:
    original = df[col].unique().tolist()
    df[col] = le.fit_transform(df[col])
    encoded = sorted(df[col].unique().tolist())
    print(f"  '{col}': {original} → {encoded}")

# ── 8. One-hot encode Wind Direction columns ─────────────────────────────────
print("\nOne-hot encoding Wind Direction columns:")
before_cols = df.shape[1]
df = pd.get_dummies(df, columns=wind_cols, dummy_na=False)
new_cols = [c for c in df.columns if any(c.startswith(w.split("_")[0] + "_Wind") or
            c.startswith("Wind Direction") for w in wind_cols)
            if c not in wind_cols]
after_cols = df.shape[1]
ohe_cols = [c for c in df.columns if c.startswith("Wind Direction_")]
print(f"  Created {len(ohe_cols)} one-hot columns: {ohe_cols}")
print(f"  Shape after OHE: {df.shape}")

# ── 9. Drop columns ──────────────────────────────────────────────────────────
drop_cols = ["Hectares", "Paddy yield(in Kg)", "yield_per_hectare", "Agriblock"]
df.drop(columns=drop_cols, inplace=True)
print(f"\nDropped columns: {drop_cols}")
print(f"Shape after drop: {df.shape}")

# ── 10. Drop highly correlated columns (r > 0.98) ───────────────────────────
X_temp = df.drop(columns=["isAboveAvg"])
corr_matrix = X_temp.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
if to_drop:
    print(f"\nDropping {len(to_drop)} highly correlated column(s) (r > 0.98):")
    for col in to_drop:
        partners = upper.index[upper[col] > 0.98].tolist()
        print(f"  '{col}' correlated with: {partners}")
    df.drop(columns=to_drop, inplace=True)
else:
    print("\nNo columns found with pairwise correlation > 0.98")
print(f"Shape after correlation drop: {df.shape}")

# ── 11. Train/test split ─────────────────────────────────────────────────────
X = df.drop(columns=["isAboveAvg"])
y = df["isAboveAvg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 12. Print final shapes ───────────────────────────────────────────────────
print("\nFinal split shapes:")
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")

# ── 13. Save to processed/ ───────────────────────────────────────────────────
PROCESSED.mkdir(exist_ok=True)
X_train.to_csv(PROCESSED / "X_train.csv", index=False)
X_test.to_csv(PROCESSED / "X_test.csv", index=False)
y_train.to_csv(PROCESSED / "y_train.csv", index=False)
y_test.to_csv(PROCESSED / "y_test.csv", index=False)
print(f"\nSaved to {PROCESSED}/")
print("  X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("\nDone.")
