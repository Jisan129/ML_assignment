import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS    = Path(__file__).parent.parent / "models"
MODELS.mkdir(exist_ok=True)

# ── 1. Load splits ────────────────────────────────────────────────────────────
X_train = pd.read_csv(PROCESSED / "X_train.csv")
X_test  = pd.read_csv(PROCESSED / "X_test.csv")
y_train = pd.read_csv(PROCESSED / "y_train.csv").squeeze()
y_test  = pd.read_csv(PROCESSED / "y_test.csv").squeeze()
print(f"Loaded — X_train: {X_train.shape}, X_test: {X_test.shape}")

feature_names = X_train.columns.tolist()

# ── 2. StandardScaler (fit on train only) ─────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. Full Logistic Regression with GridSearchCV ─────────────────────────────
print("\n" + "=" * 60)
print("FULL LOGISTIC REGRESSION (GridSearchCV)")
print("=" * 60)

param_grid = {
    "C"       : [0.01, 0.1, 1, 10, 100],
    "solver"  : ["lbfgs", "liblinear"],
    "max_iter": [200, 500, 1000],
}
grid_full = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_full.fit(X_train_sc, y_train)
lr_full = grid_full.best_estimator_

# ── 4. Best parameters ────────────────────────────────────────────────────────
print("Best parameters:")
for k, v in grid_full.best_params_.items():
    print(f"  {k:<10}: {v}")
print(f"Best CV accuracy  : {grid_full.best_score_:.4f}")

# ── 5. Accuracy ───────────────────────────────────────────────────────────────
train_acc = accuracy_score(y_train, lr_full.predict(X_train_sc))
test_acc  = accuracy_score(y_test,  lr_full.predict(X_test_sc))
print(f"Training accuracy : {train_acc:.4f}")
print(f"Test accuracy     : {test_acc:.4f}")

# ── 6. Top 10 features by absolute coefficient ────────────────────────────────
coefs = pd.Series(np.abs(lr_full.coef_[0]), index=feature_names)
top10_lr = coefs.nlargest(10)
print("\nTop 10 features by |coefficient| (full LR):")
for feat, val in top10_lr.items():
    raw_coef = lr_full.coef_[0][feature_names.index(feat)]
    print(f"  {feat:<45} coef={raw_coef:+.4f}  |coef|={val:.4f}")

# ── 7. Rebuild tuned Decision Tree to get its top-10 features ─────────────────
print("\n" + "=" * 60)
print("REDUCED LOGISTIC REGRESSION (top-10 DT features)")
print("=" * 60)

dt_best = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, min_samples_leaf=1,
    min_samples_split=2, random_state=42
)
dt_best.fit(X_train, y_train)
dt_importances = pd.Series(dt_best.feature_importances_, index=feature_names)
top10_dt_feats = dt_importances.nlargest(10).index.tolist()

print("Top-10 DT features used for reduced model:")
for i, f in enumerate(top10_dt_feats, 1):
    print(f"  {i:>2}. {f}")

# Scale reduced feature sets
X_train_red = X_train[top10_dt_feats].values
X_test_red  = X_test[top10_dt_feats].values

scaler_red = StandardScaler()
X_train_red_sc = scaler_red.fit_transform(X_train_red)
X_test_red_sc  = scaler_red.transform(X_test_red)

# ── GridSearchCV for reduced model ────────────────────────────────────────────
grid_red = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_red.fit(X_train_red_sc, y_train)
lr_red = grid_red.best_estimator_

# ── 8. Best parameters (reduced) ─────────────────────────────────────────────
print("\nBest parameters (reduced):")
for k, v in grid_red.best_params_.items():
    print(f"  {k:<10}: {v}")
print(f"Best CV accuracy  : {grid_red.best_score_:.4f}")

# ── 9. Accuracy (reduced) ─────────────────────────────────────────────────────
train_acc_r = accuracy_score(y_train, lr_red.predict(X_train_red_sc))
test_acc_r  = accuracy_score(y_test,  lr_red.predict(X_test_red_sc))
print(f"Training accuracy : {train_acc_r:.4f}")
print(f"Test accuracy     : {test_acc_r:.4f}")

# ── 10. Features used in reduced model ───────────────────────────────────────
print("\nFeatures in reduced model with their coefficients:")
for feat, coef in zip(top10_dt_feats, lr_red.coef_[0]):
    print(f"  {feat:<45} coef={coef:+.4f}")

# ── 11. ROC curves ────────────────────────────────────────────────────────────
print("\nSaving ROC curve plot...")
fig, ax = plt.subplots(figsize=(8, 6))

models_info = [
    ("Full LR (45 features)",    lr_full, X_test_sc,     "steelblue"),
    ("Reduced LR (10 features)", lr_red,  X_test_red_sc, "darkorange"),
]
for label, model, X_t, color in models_info:
    y_prob = model.predict_proba(X_t)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Full vs Reduced Logistic Regression", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(MODELS / "roc_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved models/roc_logistic_regression.png")

print("\nDone.")
