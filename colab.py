# =============================================================================
#  PADDY YIELD PREDICTION — ALL TASKS (Notebook / Colab Ready)
#  Dataset : Asm1_dataset26.csv
#  Target  : isAboveAvg  (1 = yield_per_hectare > mean, else 0)
# =============================================================================
#
#  HOW TO USE
#  ──────────
#  Google Colab:
#    1. Upload Asm1_dataset26.csv to your Colab session (Files panel on the left)
#    2. Set DATA_PATH in Cell 0 to "/content/Asm1_dataset26.csv"
#    3. Run all cells top-to-bottom (Runtime → Run all)
#
#  Local Jupyter / VS Code:
#    1. Set DATA_PATH to the path of your CSV file
#    2. Run all cells top-to-bottom
#
#  Each "# %%" marker is a separate notebook cell.
# =============================================================================
from IPython.core.display_functions import display

# %% ── Cell 0 : CONFIGURATION (edit this before running) ────────────────────

# !! Set this to the path of your CSV file !!
DATA_PATH = "Asm1_dataset26.csv"   # Colab: "/content/Asm1_dataset26.csv"

# Where to save processed splits and model images (created automatically)
import os
PROCESSED_DIR = "processed"
MODELS_DIR    = "models"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,    exist_ok=True)

print("Configuration set.")
print(f"  DATA_PATH     : {DATA_PATH}")
print(f"  PROCESSED_DIR : {PROCESSED_DIR}")
print(f"  MODELS_DIR    : {MODELS_DIR}")


# %% ── Cell 1 : INSTALL / IMPORT LIBRARIES ───────────────────────────────────

# Uncomment the line below if running on Colab and libraries are missing:
# !pip install -q scikit-learn pandas numpy matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree            import DecisionTreeClassifier, plot_tree
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, roc_curve, auc

print("All libraries imported successfully.")


# =============================================================================
#  ████████╗ █████╗ ███████╗██╗  ██╗     ██╗
#     ██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ███║
#     ██║   ███████║███████╗█████╔╝     ╚██║
#     ██║   ██╔══██║╚════██║██╔═██╗      ██║
#     ██║   ██║  ██║███████║██║  ██╗     ██║
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝
#  DATA PREPARATION & FEATURE ENGINEERING
# =============================================================================

# %% ── Cell 2 : LOAD DATASET ─────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()   # remove any leading/trailing whitespace

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>2}. {col}")


# %% ── Cell 3 : QUICK EXPLORATION ────────────────────────────────────────────

print("── Shape ──────────────────────────────────────")
print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

print("\n── First 5 rows ────────────────────────────────")
display(df.head()) if "display" in dir() else print(df.head().to_string())

print("\n── df.info() ───────────────────────────────────")
df.info()

print("\n── df.describe() ───────────────────────────────")
display(df.describe(include="all")) if "display" in dir() else print(df.describe(include="all").to_string())


# %% ── Cell 4 : REPLACE '--' IN WIND DIRECTION COLUMNS WITH NaN ──────────────

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


# %% ── Cell 5 : IMPUTE MISSING MIN TEMP COLUMNS WITH COLUMN MEAN ─────────────

min_temp_cols = [
    "Min temp_D1_D30",
    "Min temp_D31_D60",
    "Min temp_D61_D90",
    "Min temp_D91_D120",
]

print("Imputing missing Min temp values with column mean:")
for col in min_temp_cols:
    missing  = df[col].isna().sum()
    col_mean = df[col].mean()
    df[col]  = df[col].fillna(col_mean)
    print(f"  '{col}': filled {missing} NaN(s) with mean = {col_mean:.4f}")


# %% ── Cell 6 : CALCULATE yield_per_hectare & CREATE BINARY TARGET ────────────

df["yield_per_hectare"] = df["Paddy yield(in Kg)"] / df["Hectares"]

mean_yield = df["yield_per_hectare"].mean()
df["isAboveAvg"] = (df["yield_per_hectare"] > mean_yield).astype(int)

print(f"yield_per_hectare  →  min: {df['yield_per_hectare'].min():.2f} | "
      f"max: {df['yield_per_hectare'].max():.2f} | mean: {mean_yield:.2f}")

print("\nisAboveAvg value counts:")
print(df["isAboveAvg"].value_counts().to_string())
print(f"\n  Threshold: yield_per_hectare > {mean_yield:.2f} kg/ha → labelled 1")

# Plot target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(["Below Avg (0)", "Above Avg (1)"],
            df["isAboveAvg"].value_counts().sort_index(),
            color=["#E67E22", "#27AE60"], edgecolor="white", width=0.5)
axes[0].set_title("Target Distribution — isAboveAvg", fontweight="bold")
axes[0].set_ylabel("Count")
for bar, v in zip(axes[0].patches, df["isAboveAvg"].value_counts().sort_index()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(v), ha="center", fontweight="bold")

axes[1].hist(df["yield_per_hectare"], bins=30, color="#2471A3",
             edgecolor="white", alpha=0.9)
axes[1].axvline(mean_yield, color="#E67E22", lw=2, linestyle="--",
                label=f"Mean = {mean_yield:.0f}")
axes[1].set_title("Distribution of yield_per_hectare", fontweight="bold")
axes[1].set_xlabel("Yield per Hectare (kg)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Cell 7 : LABEL ENCODE — Variety, Soil Types, Nursery ──────────────────

label_cols = ["Variety", "Soil Types", "Nursery"]
le = LabelEncoder()

print("Label encoding:")
for col in label_cols:
    original = df[col].unique().tolist()
    df[col]  = le.fit_transform(df[col])
    print(f"  '{col}': {original}  →  {sorted(df[col].unique().tolist())}")


# %% ── Cell 8 : ONE-HOT ENCODE — Wind Direction columns ──────────────────────

df = pd.get_dummies(df, columns=wind_cols, dummy_na=False)

ohe_cols = [c for c in df.columns if c.startswith("Wind Direction_")]
print(f"Created {len(ohe_cols)} one-hot columns:")
for c in ohe_cols:
    print(f"  {c}")
print(f"\nShape after OHE: {df.shape}")


# %% ── Cell 9 : DROP UNNECESSARY COLUMNS ─────────────────────────────────────

drop_cols = ["Hectares", "Paddy yield(in Kg)", "yield_per_hectare", "Agriblock"]
df.drop(columns=drop_cols, inplace=True)

print(f"Dropped: {drop_cols}")
print(f"Shape after drop: {df.shape}")


# %% ── Cell 10 : DROP HIGHLY CORRELATED COLUMNS (r > 0.98) ───────────────────

X_temp      = df.drop(columns=["isAboveAvg"])
corr_matrix = X_temp.corr().abs()
upper       = corr_matrix.where(
                  np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop     = [col for col in upper.columns if any(upper[col] > 0.98)]

if to_drop:
    print(f"Dropping {len(to_drop)} highly correlated column(s) (r > 0.98):")
    for col in to_drop:
        partners = upper.index[upper[col] > 0.98].tolist()
        print(f"  '{col}'  ←→  {partners}")
    df.drop(columns=to_drop, inplace=True)
else:
    print("No columns with pairwise correlation > 0.98 found.")

print(f"\nShape after correlation drop: {df.shape}")

# Correlation heatmap (top features only, for readability)
top_feats = X_temp.drop(columns=to_drop).corr().abs()
plt.figure(figsize=(8, 6))
plt.imshow(top_feats.values[:15, :15], cmap="coolwarm", vmin=0, vmax=1)
plt.colorbar(label="Absolute Correlation")
plt.xticks(range(15), top_feats.columns[:15], rotation=45, ha="right", fontsize=7)
plt.yticks(range(15), top_feats.columns[:15], fontsize=7)
plt.title("Correlation Heatmap — First 15 Features After Pruning", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Cell 11 : TRAIN / TEST SPLIT & SAVE ───────────────────────────────────

X = df.drop(columns=["isAboveAvg"])
y = df["isAboveAvg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Final split shapes:")
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")

X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
X_test.to_csv( f"{PROCESSED_DIR}/X_test.csv",  index=False)
y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
y_test.to_csv( f"{PROCESSED_DIR}/y_test.csv",  index=False)
print(f"\nSaved splits to '{PROCESSED_DIR}/'")


# =============================================================================
#  ████████╗ █████╗ ███████╗██╗  ██╗    ██████╗
#     ██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ╚════██╗
#     ██║   ███████║███████╗█████╔╝      █████╔╝
#     ██║   ██╔══██║╚════██║██╔═██╗     ██╔═══╝
#     ██║   ██║  ██║███████║██║  ██╗    ███████╗
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚══════╝
#  DECISION TREE CLASSIFIER
# =============================================================================

# %% ── Cell 12 : LOAD PROCESSED SPLITS ───────────────────────────────────────

X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
feature_names = X_train.columns.tolist()

print(f"Loaded — X_train: {X_train.shape},  X_test: {X_test.shape}")


# %% ── Cell 13 : DEFAULT DECISION TREE ───────────────────────────────────────

dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

train_acc = accuracy_score(y_train, dt_default.predict(X_train))
test_acc  = accuracy_score(y_test,  dt_default.predict(X_test))

print("=" * 50)
print("DEFAULT DECISION TREE")
print("=" * 50)
print(f"Training accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"Test accuracy     : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"Overfit gap       : {(train_acc - test_acc)*100:.2f} pp")
print(f"Number of nodes   : {dt_default.tree_.node_count}")
print(f"Max depth         : {dt_default.get_depth()}")

# First and second split variables
tree_   = dt_default.tree_
root_f  = feature_names[tree_.feature[0]]
left_c  = tree_.children_left[0]
right_c = tree_.children_right[0]
left_f  = feature_names[tree_.feature[left_c]]  if tree_.feature[left_c]  >= 0 else "leaf"
right_f = feature_names[tree_.feature[right_c]] if tree_.feature[right_c] >= 0 else "leaf"

print(f"\n1st split variable : '{root_f}'")
print(f"2nd split — left   : '{left_f}'")
print(f"2nd split — right  : '{right_f}'")

# Top 10 feature importances
imp_default = pd.Series(dt_default.feature_importances_, index=feature_names).nlargest(10)
print("\nTop 10 feature importances (default):")
print(imp_default.to_string())


# %% ── Cell 14 : DEFAULT TREE — VISUALISATION ────────────────────────────────

fig, ax = plt.subplots(figsize=(24, 8))
plot_tree(dt_default, max_depth=3, feature_names=feature_names,
          class_names=["BelowAvg", "AboveAvg"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Default Decision Tree  (max_depth=3 view  |  full depth=16)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/tree_default.png", dpi=150, bbox_inches="tight")
plt.show()

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(9, 5))
imp_default.sort_values().plot.barh(ax=ax, color="#2471A3", edgecolor="white")
for bar, v in zip(ax.patches, imp_default.sort_values().values):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8)
ax.set_title("Top-10 Feature Importances — Default Decision Tree", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/feat_imp_default.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Cell 15 : TUNED DECISION TREE — GridSearchCV ──────────────────────────

param_grid_dt = {
    "max_depth"        : [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf" : [1, 2, 5, 10],
    "criterion"        : ["gini", "entropy"],
}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_dt.fit(X_train, y_train)
dt_tuned = grid_dt.best_estimator_

print("=" * 50)
print("TUNED DECISION TREE — GridSearchCV")
print("=" * 50)
print("Best parameters:")
for k, v in grid_dt.best_params_.items():
    print(f"  {k:<22} : {v}")

train_acc_t = accuracy_score(y_train, dt_tuned.predict(X_train))
test_acc_t  = accuracy_score(y_test,  dt_tuned.predict(X_test))

print(f"\nBest CV accuracy  : {grid_dt.best_score_:.4f}  ({grid_dt.best_score_*100:.2f}%)")
print(f"Training accuracy : {train_acc_t:.4f}  ({train_acc_t*100:.2f}%)")
print(f"Test accuracy     : {test_acc_t:.4f}  ({test_acc_t*100:.2f}%)")
print(f"Overfit gap       : {(train_acc_t - test_acc_t)*100:.2f} pp")
print(f"Number of nodes   : {dt_tuned.tree_.node_count}")
print(f"Max depth         : {dt_tuned.get_depth()}")

imp_tuned = pd.Series(dt_tuned.feature_importances_, index=feature_names).nlargest(10)
print("\nTop 10 feature importances (tuned):")
print(imp_tuned.to_string())


# %% ── Cell 16 : TUNED TREE — VISUALISATION & FEATURE IMPORTANCE ─────────────

fig, ax = plt.subplots(figsize=(24, 8))
plot_tree(dt_tuned, max_depth=3, feature_names=feature_names,
          class_names=["BelowAvg", "AboveAvg"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Tuned Decision Tree  (max_depth=3 view  |  full depth=5)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/tree_tuned.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
imp_tuned.sort_values().plot.barh(ax=ax, color="#E67E22", edgecolor="white")
for bar, v in zip(ax.patches, imp_tuned.sort_values().values):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8)
ax.set_title("Top-10 Feature Importances — Tuned Decision Tree", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/feat_imp_tuned.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Cell 17 : DECISION TREE — ROC CURVES ──────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

for label, model, color in [
    ("Default DT", dt_default, "steelblue"),
    ("Tuned DT",   dt_tuned,   "darkorange"),
]:
    y_prob   = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc  = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label}  (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — Default vs Tuned Decision Tree", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/roc_decision_trees.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Cell 18 : DEFAULT vs TUNED — SIDE-BY-SIDE SUMMARY ─────────────────────

summary_dt = pd.DataFrame({
    "Metric"   : ["Training Acc", "Test Acc", "Overfit Gap", "Nodes", "Depth"],
    "Default DT": [f"{train_acc*100:.2f}%", f"{test_acc*100:.2f}%",
                   f"{(train_acc-test_acc)*100:.2f} pp", "545", "16"],
    "Tuned DT" : [f"{train_acc_t*100:.2f}%", f"{test_acc_t*100:.2f}%",
                  f"{(train_acc_t-test_acc_t)*100:.2f} pp", "39", "5"],
}).set_index("Metric")

print(summary_dt.to_string())
display(summary_dt) if "display" in dir() else None


# =============================================================================
#  ████████╗ █████╗ ███████╗██╗  ██╗    ██████╗
#     ██╔══╝██╔══██╗██╔════╝██║ ██╔╝    ╚════██╗
#     ██║   ███████║███████╗█████╔╝      ██████╗
#     ██║   ██╔══██║╚════██║██╔═██╗     ╚═══███╗
#     ██║   ██║  ██║███████║██║  ██╗    ██████╔╝
#     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═════╝
#  LOGISTIC REGRESSION
# =============================================================================

# %% ── Cell 19 : SCALE FEATURES ──────────────────────────────────────────────

# Reload to ensure clean state (safe to skip if running sequentially)
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
feature_names = X_train.columns.tolist()

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit on train only
X_test_sc  = scaler.transform(X_test)        # apply same scale to test

print("StandardScaler applied (fit on X_train only).")
print(f"X_train_sc shape: {X_train_sc.shape}")
print(f"X_test_sc  shape: {X_test_sc.shape}")


# %% ── Cell 20 : FULL LOGISTIC REGRESSION — GridSearchCV ─────────────────────

param_grid_lr = {
    "C"       : [0.01, 0.1, 1, 10, 100],
    "solver"  : ["lbfgs", "liblinear"],
    "max_iter": [200, 500, 1000],
}

grid_lr_full = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_lr,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_lr_full.fit(X_train_sc, y_train)
lr_full = grid_lr_full.best_estimator_

train_acc_lr = accuracy_score(y_train, lr_full.predict(X_train_sc))
test_acc_lr  = accuracy_score(y_test,  lr_full.predict(X_test_sc))

print("=" * 50)
print("FULL LOGISTIC REGRESSION (45 features)")
print("=" * 50)
print("Best parameters:")
for k, v in grid_lr_full.best_params_.items():
    print(f"  {k:<10} : {v}")
print(f"\nBest CV accuracy  : {grid_lr_full.best_score_:.4f}  ({grid_lr_full.best_score_*100:.2f}%)")
print(f"Training accuracy : {train_acc_lr:.4f}  ({train_acc_lr*100:.2f}%)")
print(f"Test accuracy     : {test_acc_lr:.4f}  ({test_acc_lr*100:.2f}%)")
print(f"Overfit gap       : {(train_acc_lr - test_acc_lr)*100:.2f} pp")

coefs_full = pd.Series(lr_full.coef_[0], index=feature_names)
top10_full = coefs_full.abs().nlargest(10)
print("\nTop 10 features by |coefficient|:")
for feat, abs_val in top10_full.items():
    print(f"  {feat:<45}  coef = {coefs_full[feat]:+.4f}   |coef| = {abs_val:.4f}")


# %% ── Cell 21 : FULL LR — COEFFICIENT CHART ─────────────────────────────────

signed = coefs_full[top10_full.index].sort_values()
colors = ["#27AE60" if v > 0 else "#C0392B" for v in signed.values]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(signed.index, signed.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", lw=0.8, linestyle="--")
for bar, v in zip(bars, signed.values):
    ha = "left" if v >= 0 else "right"
    ax.text(v + (0.005 if v >= 0 else -0.005),
            bar.get_y() + bar.get_height()/2,
            f"{v:+.4f}", va="center", ha=ha, fontsize=8)
ax.set_title("Top-10 Feature Coefficients — Full Logistic Regression\n"
             "(green = positive → increases P(AboveAvg), red = negative)",
             fontweight="bold")
ax.set_xlabel("Coefficient Value (scaled features)")
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/coef_full_lr.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Cell 22 : REDUCED LR — TOP-10 FEATURES FROM TUNED DECISION TREE ───────

# Re-fit tuned DT with the best params found in Task 2
dt_for_lr = DecisionTreeClassifier(
    criterion="entropy", max_depth=5,
    min_samples_leaf=1, min_samples_split=2, random_state=42
)
dt_for_lr.fit(X_train, y_train)

top10_dt_feats = (pd.Series(dt_for_lr.feature_importances_, index=feature_names)
                  .nlargest(10).index.tolist())

print("Top-10 DT features used for reduced Logistic Regression:")
for i, f in enumerate(top10_dt_feats, 1):
    print(f"  {i:>2}. {f}")

# Scale the reduced feature sets
X_train_red = X_train[top10_dt_feats].values
X_test_red  = X_test[top10_dt_feats].values
scaler_red      = StandardScaler()
X_train_red_sc  = scaler_red.fit_transform(X_train_red)
X_test_red_sc   = scaler_red.transform(X_test_red)


# %% ── Cell 23 : REDUCED LOGISTIC REGRESSION — GridSearchCV ──────────────────

grid_lr_red = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_lr,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_lr_red.fit(X_train_red_sc, y_train)
lr_red = grid_lr_red.best_estimator_

train_acc_lr_r = accuracy_score(y_train, lr_red.predict(X_train_red_sc))
test_acc_lr_r  = accuracy_score(y_test,  lr_red.predict(X_test_red_sc))

print("=" * 50)
print("REDUCED LOGISTIC REGRESSION (10 features)")
print("=" * 50)
print("Best parameters:")
for k, v in grid_lr_red.best_params_.items():
    print(f"  {k:<10} : {v}")
print(f"\nBest CV accuracy  : {grid_lr_red.best_score_:.4f}  ({grid_lr_red.best_score_*100:.2f}%)")
print(f"Training accuracy : {train_acc_lr_r:.4f}  ({train_acc_lr_r*100:.2f}%)")
print(f"Test accuracy     : {test_acc_lr_r:.4f}  ({test_acc_lr_r*100:.2f}%)")
print(f"Overfit gap       : {(train_acc_lr_r - test_acc_lr_r)*100:.2f} pp")

print("\nFeatures and their coefficients in the reduced model:")
for feat, coef in zip(top10_dt_feats, lr_red.coef_[0]):
    print(f"  {feat:<45}  coef = {coef:+.4f}")


# %% ── Cell 24 : LOGISTIC REGRESSION — ROC CURVES ────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

for label, model, X_t, color in [
    ("Full LR (45 features)",    lr_full, X_test_sc,    "steelblue"),
    ("Reduced LR (10 features)", lr_red,  X_test_red_sc,"darkorange"),
]:
    y_prob   = model.predict_proba(X_t)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc  = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label}  (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — Full vs Reduced Logistic Regression",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/roc_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.show()


# %% ── Cell 25 : FULL vs REDUCED LR — SUMMARY TABLE ─────────────────────────

summary_lr = pd.DataFrame({
    "Metric"    : ["Best CV Acc", "Training Acc", "Test Acc", "Overfit Gap", "Features"],
    "Full LR"   : [f"{grid_lr_full.best_score_*100:.2f}%",
                   f"{train_acc_lr*100:.2f}%", f"{test_acc_lr*100:.2f}%",
                   f"{(train_acc_lr - test_acc_lr)*100:.2f} pp", "45"],
    "Reduced LR": [f"{grid_lr_red.best_score_*100:.2f}%",
                   f"{train_acc_lr_r*100:.2f}%", f"{test_acc_lr_r*100:.2f}%",
                   f"{(train_acc_lr_r - test_acc_lr_r)*100:.2f} pp", "10"],
}).set_index("Metric")

print(summary_lr.to_string())
display(summary_lr) if "display" in dir() else None


# =============================================================================
#  OVERALL MODEL COMPARISON
# =============================================================================

# %% ── Cell 26 : ALL MODELS — FINAL COMPARISON ───────────────────────────────

fpr_dd, tpr_dd, _ = roc_curve(y_test, dt_default.predict_proba(X_test)[:,1])
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_tuned.predict_proba(X_test)[:,1])
fpr_lf, tpr_lf, _ = roc_curve(y_test, lr_full.predict_proba(X_test_sc)[:,1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_red.predict_proba(X_test_red_sc)[:,1])

models_summary = {
    "Model"        : ["Default DT", "Tuned DT", "Full LR", "Reduced LR"],
    "Train Acc (%)" : [round(train_acc*100,2), round(train_acc_t*100,2),
                       round(train_acc_lr*100,2), round(train_acc_lr_r*100,2)],
    "Test Acc (%)"  : [round(test_acc*100,2), round(test_acc_t*100,2),
                       round(test_acc_lr*100,2), round(test_acc_lr_r*100,2)],
    "AUC"          : [round(auc(fpr_dd,tpr_dd),4), round(auc(fpr_dt,tpr_dt),4),
                      round(auc(fpr_lf,tpr_lf),4), round(auc(fpr_lr,tpr_lr),4)],
    "Features"     : [45, 45, 45, 10],
    "Overfit Gap"  : [round((train_acc-test_acc)*100,2),
                      round((train_acc_t-test_acc_t)*100,2),
                      round((train_acc_lr-test_acc_lr)*100,2),
                      round((train_acc_lr_r-test_acc_lr_r)*100,2)],
}
summary_all = pd.DataFrame(models_summary).set_index("Model")
print("=" * 65)
print("ALL MODELS — FINAL COMPARISON")
print("=" * 65)
print(summary_all.to_string())
display(summary_all) if "display" in dir() else None

# ── Grouped bar chart ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x   = np.arange(4)
w   = 0.35
lbl = ["Default DT", "Tuned DT", "Full LR", "Reduced LR"]

# Train vs Test
axes[0].bar(x - w/2, summary_all["Train Acc (%)"], w,
            label="Train", color="#2471A3", edgecolor="white")
axes[0].bar(x + w/2, summary_all["Test Acc (%)"],  w,
            label="Test",  color="#27AE60", edgecolor="white")
for i, (tr, te) in enumerate(zip(summary_all["Train Acc (%)"],
                                  summary_all["Test Acc (%)"])):
    axes[0].text(i - w/2, tr + 0.2, f"{tr}", ha="center", fontsize=8, fontweight="bold")
    axes[0].text(i + w/2, te + 0.2, f"{te}", ha="center", fontsize=8, fontweight="bold")
axes[0].set_xticks(x); axes[0].set_xticklabels(lbl, fontsize=9)
axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(78, 97)
axes[0].set_title("Train vs Test Accuracy — All Models", fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

# AUC
auc_vals   = summary_all["AUC"].values
auc_colors = ["#C0392B", "#27AE60", "#2471A3", "#E67E22"]
bars_auc   = axes[1].bar(lbl, auc_vals, color=auc_colors, edgecolor="white")
for bar, v in zip(bars_auc, auc_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
axes[1].set_ylim(0.75, 1.0)
axes[1].set_xticklabels(lbl, fontsize=9, rotation=10)
axes[1].set_ylabel("AUC")
axes[1].set_title("AUC Comparison — All Models", fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{MODELS_DIR}/overall_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ All tasks complete!")
print(f"  Charts saved in '{MODELS_DIR}/'")
print(f"  Splits  saved in '{PROCESSED_DIR}/'")
