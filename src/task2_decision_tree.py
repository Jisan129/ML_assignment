import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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

# ── 2. Default Decision Tree ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DEFAULT DECISION TREE")
print("=" * 60)
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

# ── 3. Accuracy ───────────────────────────────────────────────────────────────
train_acc = accuracy_score(y_train, dt_default.predict(X_train))
test_acc  = accuracy_score(y_test,  dt_default.predict(X_test))
print(f"Training accuracy : {train_acc:.4f}")
print(f"Test accuracy     : {test_acc:.4f}")

# ── 4. Nodes and depth ────────────────────────────────────────────────────────
print(f"Number of nodes   : {dt_default.tree_.node_count}")
print(f"Max depth         : {dt_default.get_depth()}")

# ── 5. First and second split variables ──────────────────────────────────────
feature_names = X_train.columns.tolist()
tree_ = dt_default.tree_

def get_split_feature(tree, node_id, feature_names):
    return feature_names[tree.feature[node_id]]

root_feature = get_split_feature(tree_, 0, feature_names)
left_child   = tree_.children_left[0]
right_child  = tree_.children_right[0]

print(f"1st split variable: '{root_feature}'")

left_feat  = get_split_feature(tree_, left_child,  feature_names) if tree_.feature[left_child]  >= 0 else "leaf"
right_feat = get_split_feature(tree_, right_child, feature_names) if tree_.feature[right_child] >= 0 else "leaf"
print(f"2nd split variable: left branch  → '{left_feat}'")
print(f"                    right branch → '{right_feat}'")

# ── 6. Top 10 important features ─────────────────────────────────────────────
importances = pd.Series(dt_default.feature_importances_, index=feature_names)
top10_default = importances.nlargest(10)
print("\nTop 10 feature importances (default):")
for feat, imp in top10_default.items():
    print(f"  {feat:<45} {imp:.6f}")

# ── 7. Tuned Decision Tree with GridSearchCV ──────────────────────────────────
print("\n" + "=" * 60)
print("TUNED DECISION TREE (GridSearchCV)")
print("=" * 60)
param_grid = {
    "max_depth"        : [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf" : [1, 2, 5, 10],
    "criterion"        : ["gini", "entropy"],
}
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)
dt_tuned = grid_search.best_estimator_

# ── 8. Best parameters ────────────────────────────────────────────────────────
print("Best parameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k:<22}: {v}")
print(f"Best CV accuracy  : {grid_search.best_score_:.4f}")

# ── 9. Tuned accuracy ─────────────────────────────────────────────────────────
train_acc_t = accuracy_score(y_train, dt_tuned.predict(X_train))
test_acc_t  = accuracy_score(y_test,  dt_tuned.predict(X_test))
print(f"Training accuracy : {train_acc_t:.4f}")
print(f"Test accuracy     : {test_acc_t:.4f}")

# ── 10. Nodes and depth (tuned) ───────────────────────────────────────────────
print(f"Number of nodes   : {dt_tuned.tree_.node_count}")
print(f"Max depth         : {dt_tuned.get_depth()}")

# ── 11. Top 10 important features (tuned) ────────────────────────────────────
importances_t = pd.Series(dt_tuned.feature_importances_, index=feature_names)
top10_tuned   = importances_t.nlargest(10)
print("\nTop 10 feature importances (tuned):")
for feat, imp in top10_tuned.items():
    print(f"  {feat:<45} {imp:.6f}")

# ── 12. Visualise both trees (max_depth=3) ───────────────────────────────────
print("\nSaving tree visualisations...")
for label, model, fname in [
    ("Default", dt_default, "tree_default.png"),
    ("Tuned",   dt_tuned,   "tree_tuned.png"),
]:
    fig, ax = plt.subplots(figsize=(24, 8))
    plot_tree(
        model,
        max_depth=3,
        feature_names=feature_names,
        class_names=["BelowAvg", "AboveAvg"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"{label} Decision Tree (max_depth=3 view)", fontsize=14)
    fig.tight_layout()
    fig.savefig(MODELS / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved models/{fname}")

# ── 13. ROC curves ────────────────────────────────────────────────────────────
print("Saving ROC curve plot...")
fig, ax = plt.subplots(figsize=(8, 6))

for label, model, color in [
    ("Default DT", dt_default, "steelblue"),
    ("Tuned DT",   dt_tuned,   "darkorange"),
]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Default vs Tuned Decision Tree", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(MODELS / "roc_decision_trees.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved models/roc_decision_trees.png")

print("\nDone.")
