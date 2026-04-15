"""
Paddy Yield Prediction — PDF Report Generator
Covers Task 1 (Data Prep), Task 2 (Decision Tree), Task 3 (Logistic Regression)
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS    = BASE / "models"
RAW       = BASE / "data" / "raw"
OUT_PDF   = BASE / "models" / "Paddy_Yield_Report.pdf"

# ─── Palette ──────────────────────────────────────────────────────────────────
C_NAVY    = "#1B3A5C"
C_BLUE    = "#2471A3"
C_SKY     = "#AED6F1"
C_GREEN   = "#1E8449"
C_ORANGE  = "#D35400"
C_RED     = "#C0392B"
C_GRAY    = "#F2F4F4"
C_LGRAY   = "#E8EAED"
C_LINE    = "#D0D3D4"
C_DARK    = "#1C2833"
C_WHITE   = "#FFFFFF"
C_TEAL    = "#0E6655"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ═════════════════════════════════════════════════════════════════════════════
# Low-level drawing helpers
# ═════════════════════════════════════════════════════════════════════════════

def new_page(pdf, w=8.27, h=11.69):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(C_WHITE)
    return fig

def save(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def band(fig, title, subtitle="", y=0.955, h=0.055, bg=C_NAVY, fg=C_WHITE):
    ax = fig.add_axes([0, y, 1, h])
    ax.set_facecolor(bg); ax.set_xlim(0,1); ax.set_ylim(0,1)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.018, 0.5, title, color=fg, fontsize=13, fontweight="bold", va="center")
    if subtitle:
        ax.text(0.982, 0.5, subtitle, color=C_SKY, fontsize=9, va="center", ha="right")
    return ax

def sub_band(fig, text, left, bottom, width, height=0.028, bg=C_BLUE):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor(bg); ax.set_xlim(0,1); ax.set_ylim(0,1)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.015, 0.5, text, color=C_WHITE, fontsize=9.5,
            fontweight="bold", va="center")

def code_box(fig, lines, left, bottom, width, height, fs=6.8):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor("#1E1E2E"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    for s in ax.spines.values(): s.set_color("#444"), s.set_linewidth(0.6)
    ax.set_xticks([]); ax.set_yticks([])
    # line numbers + syntax-ish colouring (keywords in teal, strings in orange)
    KEYWORDS = {"import","from","as","def","class","return","for","in","if",
                "else","and","or","not","True","False","None","print","with"}
    n = len(lines)
    step = min(1.0 / (n + 1), 0.065)
    y0   = 1 - step * 0.7
    for i, raw in enumerate(lines):
        y = y0 - i * step
        if y < 0.02: break
        # line number
        ax.text(0.008, y, f"{i+1:>2}", color="#555577", fontsize=fs-1,
                fontfamily="monospace", va="center")
        stripped = raw.lstrip()
        indent   = len(raw) - len(stripped)
        col = "#A9DC76"  # default green (identifiers/values)
        if stripped.startswith("#"):        col = "#727072"
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            col = "#FFD866"
        else:
            first_word = stripped.split("(")[0].split(" ")[0].split("=")[0]
            if first_word in KEYWORDS:      col = "#78DCE8"
            elif "=" in stripped and "==" not in stripped: col = "#FFFFFF"
        ax.text(0.055 + indent * 0.008, y, raw, color=col, fontsize=fs,
                fontfamily="monospace", va="center")
    return ax

def grid_table(fig, left, bottom, width, height,
               headers, rows, col_w=None, hdr_bg=C_NAVY, fs=7.8):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_facecolor(C_WHITE)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    nc = len(headers)
    nr = len(rows)
    if col_w is None: col_w = [1/nc]*nc
    rh = 1/(nr+1)
    # header
    x = 0
    for j,(h,cw) in enumerate(zip(headers,col_w)):
        r = patches.Rectangle((x, 1-rh), cw, rh,
                               facecolor=hdr_bg, edgecolor=C_WHITE, lw=0.4)
        ax.add_patch(r)
        ax.text(x+cw/2, 1-rh/2, h, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=C_WHITE)
        x += cw
    # rows
    for i,row in enumerate(rows):
        bg = C_LGRAY if i%2==0 else C_WHITE
        x = 0
        for j,(cell,cw) in enumerate(zip(row,col_w)):
            r = patches.Rectangle((x, 1-(i+2)*rh), cw, rh,
                                   facecolor=bg, edgecolor=C_LINE, lw=0.3)
            ax.add_patch(r)
            ax.text(x+cw/2, 1-(i+1.5)*rh, str(cell), ha="center", va="center",
                    fontsize=fs, color=C_DARK)
            x += cw
    return ax

def body_text(fig, lines, left, bottom, width, height, fs=8.5, gap=1.6):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor(C_WHITE)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    n   = len(lines)
    lh  = min(1/n, 0.12)
    y   = 1 - lh*0.6
    for line in lines:
        if y < 0: break
        indent = "   " if line.startswith("•") else ""
        color  = C_DARK
        fw     = "normal"
        if line.startswith("##"):
            line  = line[2:].strip()
            color = C_BLUE; fw = "bold"; fs_use = fs+1
        elif line.startswith("#"):
            line  = line[1:].strip()
            color = C_NAVY; fw = "bold"; fs_use = fs+0.5
        else:
            fs_use = fs
        ax.text(0.01, y, indent+line, color=color, fontsize=fs_use,
                fontweight=fw, va="center", wrap=False)
        y -= lh * gap
    return ax

def img_ax(fig, left, bottom, width, height, path):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(C_LINE), s.set_linewidth(0.5)
    try:
        img = mpimg.imread(str(path))
        ax.imshow(img, aspect="auto")
    except Exception as e:
        ax.text(0.5, 0.5, f"[Image not found]\n{path.name}",
                ha="center", va="center", color="gray", fontsize=8)
    return ax

def footer(fig, page_num, total):
    ax = fig.add_axes([0, 0, 1, 0.025])
    ax.set_facecolor(C_LGRAY); ax.set_xlim(0,1); ax.set_ylim(0,1)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5,  0.5, "Paddy Yield Prediction — ML Project Report",
            ha="center", va="center", color="#7F8C8D", fontsize=7.5)
    ax.text(0.985, 0.5, f"Page {page_num} of {total}",
            ha="right",  va="center", color="#7F8C8D", fontsize=7.5)

# ═════════════════════════════════════════════════════════════════════════════
# Re-derive results for charts (deterministic, same random_state)
# ═════════════════════════════════════════════════════════════════════════════
print("Re-deriving model results for charts…")

X_train = pd.read_csv(PROCESSED/"X_train.csv")
X_test  = pd.read_csv(PROCESSED/"X_test.csv")
y_train = pd.read_csv(PROCESSED/"y_train.csv").squeeze()
y_test  = pd.read_csv(PROCESSED/"y_test.csv").squeeze()
feat    = X_train.columns.tolist()

# Decision Trees
dt_def = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
dt_tun = DecisionTreeClassifier(criterion="entropy", max_depth=5,
                                 min_samples_leaf=1, min_samples_split=2,
                                 random_state=42).fit(X_train, y_train)

# Logistic Regression
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_train)
Xte_sc = scaler.transform(X_test)
lr_full = LogisticRegression(C=0.01, solver="liblinear", max_iter=200,
                              random_state=42).fit(Xtr_sc, y_train)

imp_dt = pd.Series(dt_tun.feature_importances_, index=feat)
top10_dt = imp_dt.nlargest(10).index.tolist()
scaler_r = StandardScaler()
Xtr_r = scaler_r.fit_transform(X_train[top10_dt])
Xte_r = scaler_r.transform(X_test[top10_dt])
lr_red = LogisticRegression(C=0.01, solver="liblinear", max_iter=200,
                             random_state=42).fit(Xtr_r, y_train)

print("Done. Building PDF…")

TOTAL_PAGES = 12

# ═════════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ═════════════════════════════════════════════════════════════════════════════
with PdfPages(OUT_PDF) as pdf:

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1 — TITLE
    # ──────────────────────────────────────────────────────────────────────────
    pg = 1
    fig = new_page(pdf)

    # Navy top strip
    ax_top = fig.add_axes([0, 0.82, 1, 0.18])
    ax_top.set_facecolor(C_NAVY); ax_top.set_xlim(0,1); ax_top.set_ylim(0,1)
    for s in ax_top.spines.values(): s.set_visible(False)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.text(0.5, 0.72, "🌾  Paddy Yield Prediction", ha="center", va="center",
                fontsize=22, fontweight="bold", color=C_WHITE)
    ax_top.text(0.5, 0.30, "Machine Learning Project Report", ha="center", va="center",
                fontsize=13, color=C_SKY)

    # Accent bar
    ax_bar = fig.add_axes([0.1, 0.795, 0.8, 0.008])
    ax_bar.set_facecolor(C_ORANGE)
    for s in ax_bar.spines.values(): s.set_visible(False)
    ax_bar.set_xticks([]); ax_bar.set_yticks([])

    # Info cards
    cards = [
        ("Dataset",   "Asm1_dataset26.csv\n2,789 rows · 45 features"),
        ("Target",    "isAboveAvg\n(Binary classification)"),
        ("Tasks",     "3 tasks completed\nDT · LR · Data Prep"),
        ("Tools",     "Python · scikit-learn\npandas · matplotlib"),
    ]
    cw, ch = 0.18, 0.12
    gap = (1 - 4*cw - 0.04) / 5
    for i,(title_c,body_c) in enumerate(cards):
        lft = gap*(i+1) + cw*i + 0.02
        ax_c = fig.add_axes([lft, 0.63, cw, ch])
        ax_c.set_facecolor(C_LGRAY)
        ax_c.add_patch(patches.FancyBboxPatch((0,0),1,1,
            boxstyle="round,pad=0.03", facecolor=C_LGRAY,
            edgecolor=C_BLUE, lw=1.2))
        for s in ax_c.spines.values(): s.set_visible(False)
        ax_c.set_xlim(0,1); ax_c.set_ylim(0,1)
        ax_c.set_xticks([]); ax_c.set_yticks([])
        ax_c.text(0.5, 0.72, title_c, ha="center", va="center",
                  fontsize=8.5, fontweight="bold", color=C_NAVY)
        ax_c.text(0.5, 0.35, body_c, ha="center", va="center",
                  fontsize=7.5, color=C_DARK, linespacing=1.5)

    # TOC
    body_text(fig, [
        "## Table of Contents",
        "",
        "  1.  Task 1 — Data Preparation & Feature Engineering   (Pages 2 – 4)",
        "  2.  Task 2 — Decision Tree Classifier                  (Pages 5 – 8)",
        "  3.  Task 3 — Logistic Regression                       (Pages 9 – 11)",
        "  4.  Overall Model Comparison                           (Page 12)",
    ], 0.08, 0.35, 0.84, 0.24, fs=9.5, gap=1.8)

    # Project summary
    body_text(fig, [
        "## Project Overview",
        "",
        ("• This report documents a supervised binary classification pipeline built on a paddy "
         "cultivation dataset from Tamil Nadu, India."),
        ("• The goal is to predict whether a given farm plot will achieve above-average "
         "yield per hectare, enabling data-driven agronomic decisions."),
        ("• Three tasks were completed: thorough data preparation, a fully tuned Decision "
         "Tree, and a Logistic Regression with both full and reduced feature sets."),
    ], 0.08, 0.09, 0.84, 0.24, fs=9, gap=2.0)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 2 — TASK 1 : Dataset Exploration
    # ──────────────────────────────────────────────────────────────────────────
    pg = 2
    fig = new_page(pdf)
    band(fig, "Task 1 — Data Preparation & Feature Engineering",
         "Dataset Exploration", bg=C_TEAL)

    sub_band(fig, "1.1  Raw Dataset Overview", 0.04, 0.875, 0.92)
    grid_table(fig, 0.04, 0.76, 0.92, 0.105,
               ["Property", "Value"],
               [["File", "Asm1_dataset26.csv"],
                ["Shape", "2,789 rows × 45 columns"],
                ["Numeric columns (int64 + float64)", "37"],
                ["Categorical columns (object/str)", "8"],
                ["Duplicate rows", "0"],
                ["Rows with any missing value", "108  (Min temp columns only)"]],
               col_w=[0.55,0.45], hdr_bg=C_TEAL)

    sub_band(fig, "1.2  Categorical Columns — Unique Values", 0.04, 0.715, 0.92)
    grid_table(fig, 0.04, 0.595, 0.92, 0.115,
               ["Column", "Unique Values", "Most Frequent", "Count"],
               [["Agriblock",               "6",  "Sankarapuram", "605"],
                ["Variety",                 "3",  "ponmani",      "1,061"],
                ["Soil Types",              "2",  "clay",         "1,521"],
                ["Nursery",                 "2",  "dry",          "1,540"],
                ["Wind Direction_D1_D30",   "7",  "--  (missing)","691"],
                ["Wind Direction_D31_D60",  "6",  "--  (missing)","691"],
                ["Wind Direction_D61_D90",  "6",  "--  (missing)","691"],
                ["Wind Direction_D91_D120", "7",  "--  (missing)","691"]],
               col_w=[0.38,0.18,0.27,0.17], hdr_bg=C_TEAL)

    sub_band(fig, "1.3  Missing Values Detail", 0.04, 0.550, 0.92)
    grid_table(fig, 0.04, 0.445, 0.92, 0.100,
               ["Column", "Missing", "% Missing", "Imputation Strategy"],
               [["Min temp_D1_D30",   "108","3.87 %","Column mean (19.33 °C)"],
                ["Min temp_D31_D60",  "108","3.87 %","Column mean (17.14 °C)"],
                ["Min temp_D61_D90",  "108","3.87 %","Column mean (16.68 °C)"],
                ["Min temp_D91_D120", "108","3.87 %","Column mean (16.18 °C)"],
                ["Wind Dir columns",  "691 each","24.77 %","Replace '--' → NaN, then OHE (NaN = all-zero row)"]],
               col_w=[0.30,0.13,0.15,0.42], hdr_bg=C_TEAL)

    # Target distribution bar chart
    ax_bar2 = fig.add_axes([0.08, 0.12, 0.38, 0.28])
    ax_bar2.set_facecolor(C_GRAY)
    ax_bar2.bar(["Below Avg (0)", "Above Avg (1)"], [1415, 1374],
                color=[C_ORANGE, C_GREEN], width=0.5, edgecolor=C_WHITE, lw=1.2)
    ax_bar2.set_title("Target Distribution — isAboveAvg", fontsize=9,
                      fontweight="bold", color=C_NAVY, pad=6)
    ax_bar2.set_ylabel("Count", fontsize=8)
    for bar, v in zip(ax_bar2.patches, [1415, 1374]):
        ax_bar2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                     str(v), ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    ax_bar2.set_ylim(0, 1600)
    ax_bar2.tick_params(labelsize=8)
    ax_bar2.spines["left"].set_color(C_LINE)
    ax_bar2.spines["bottom"].set_color(C_LINE)

    # Yield per hectare histogram
    df_raw = pd.read_csv(RAW/"Asm1_dataset26.csv")
    df_raw.columns = df_raw.columns.str.strip()
    yph = df_raw["Paddy yield(in Kg)"] / df_raw["Hectares"]
    ax_hist = fig.add_axes([0.56, 0.12, 0.38, 0.28])
    ax_hist.set_facecolor(C_GRAY)
    ax_hist.hist(yph, bins=30, color=C_BLUE, edgecolor=C_WHITE, lw=0.6, alpha=0.9)
    ax_hist.axvline(yph.mean(), color=C_ORANGE, lw=1.8, linestyle="--",
                    label=f"Mean = {yph.mean():.0f}")
    ax_hist.set_title("Distribution of yield_per_hectare", fontsize=9,
                      fontweight="bold", color=C_NAVY, pad=6)
    ax_hist.set_xlabel("Yield per Hectare (kg)", fontsize=8)
    ax_hist.set_ylabel("Frequency", fontsize=8)
    ax_hist.legend(fontsize=8)
    ax_hist.tick_params(labelsize=8)
    ax_hist.spines["left"].set_color(C_LINE)
    ax_hist.spines["bottom"].set_color(C_LINE)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 3 — TASK 1 : Preprocessing Steps + Code
    # ──────────────────────────────────────────────────────────────────────────
    pg = 3
    fig = new_page(pdf)
    band(fig, "Task 1 — Data Preparation & Feature Engineering",
         "Preprocessing Pipeline", bg=C_TEAL)

    sub_band(fig, "1.4  Preprocessing Pipeline — Steps", 0.04, 0.875, 0.92)
    grid_table(fig, 0.04, 0.685, 0.92, 0.185,
               ["Step", "Action", "Detail"],
               [["1","Replace '--' with NaN","All 4 Wind Direction columns · 691 values each"],
                ["2","Impute Min temp NaNs","Fill 108 missing per column with column mean"],
                ["3","Derive yield_per_hectare","Paddy yield(in Kg) ÷ Hectares"],
                ["4","Create binary target","isAboveAvg = 1 if yield_per_hectare > mean (5,990.53)"],
                ["5","Label encode","Variety (3 classes) · Soil Types (2) · Nursery (2)"],
                ["6","One-hot encode","Wind Direction × 4 → 22 binary columns"],
                ["7","Drop columns","Hectares · Paddy yield · yield_per_hectare · Agriblock"],
                ["8","Drop correlated (r>0.98)","15 columns removed (dosage group + DAI mirrors)"],
                ["9","Train/Test split","80/20 · random_state=42 → 2,231 train · 558 test"]],
               col_w=[0.06,0.30,0.64], hdr_bg=C_TEAL)

    sub_band(fig, "1.5  Correlated Columns Removed (r > 0.98)", 0.04, 0.640, 0.92)
    grid_table(fig, 0.04, 0.470, 0.92, 0.165,
               ["Dropped Column", "Highly Correlated With"],
               [["LP_Mainfield(in Tonnes)",  "Seedrate(in Kg)"],
                ["Nursery area (Cents)",      "Seedrate, LP_Mainfield"],
                ["LP_nurseryarea(in Tonnes)", "Seedrate, LP_Mainfield, Nursery area"],
                ["DAP_20days",               "Seedrate + entire dosage group"],
                ["Weed28D_thiobencarb",      "Seedrate + entire dosage group"],
                ["Urea_40Days",              "Seedrate + entire dosage group"],
                ["Potassh_50Days",           "Seedrate + entire dosage group"],
                ["Micronutrients_70Days",    "Seedrate + entire dosage group"],
                ["Pest_60Day(in ml)",        "Seedrate + entire dosage group"],
                ["30DAI(in mm)",             "30DRain( in mm)"],
                ["30_50DAI(in mm)",          "30_50DRain( in mm)"],
                ["51_70AI(in mm)",           "51_70DRain(in mm)"],
                ["71_105DRain(in mm)",       "30_50DRain"],
                ["71_105DAI(in mm)",         "30_50DRain · 71_105DRain"],
                ["Min temp_D61_D90",         "Min temp_D1_D30"]],
               col_w=[0.44,0.56], hdr_bg=C_TEAL)

    sub_band(fig, "1.6  Key Code Snippet", 0.04, 0.425, 0.92)
    code_box(fig, [
        "df.columns = df.columns.str.strip()                    # fix trailing whitespace",
        "df[wind_cols] = df[wind_cols].replace('--', np.nan)    # step 2",
        "df[min_temp_cols] = df[min_temp_cols].fillna(df[min_temp_cols].mean())  # step 3",
        "df['yield_per_hectare'] = df['Paddy yield(in Kg)'] / df['Hectares']",
        "df['isAboveAvg'] = (df['yield_per_hectare'] > df['yield_per_hectare'].mean()).astype(int)",
        "# Label encode",
        "for col in ['Variety','Soil Types','Nursery']:",
        "    df[col] = LabelEncoder().fit_transform(df[col])",
        "# One-hot encode wind directions (NaN rows become all-zero)",
        "df = pd.get_dummies(df, columns=wind_cols, dummy_na=False)",
        "# Drop high-correlation features",
        "upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, bool), k=1))",
        "to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]",
        "df.drop(columns=to_drop, inplace=True)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
    ], 0.04, 0.09, 0.92, 0.325, fs=7.2)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 4 — TASK 1 : Final shape + Label encoding summary
    # ──────────────────────────────────────────────────────────────────────────
    pg = 4
    fig = new_page(pdf)
    band(fig, "Task 1 — Data Preparation & Feature Engineering",
         "Final Processed Dataset", bg=C_TEAL)

    sub_band(fig, "1.7  Final Train / Test Split", 0.04, 0.875, 0.92)
    grid_table(fig, 0.04, 0.780, 0.92, 0.090,
               ["Split", "Rows", "Features", "% of Total"],
               [["X_train", "2,231", "45", "80 %"],
                ["X_test",  "558",   "45", "20 %"],
                ["y_train", "2,231", "1 (isAboveAvg)", "—"],
                ["y_test",  "558",   "1 (isAboveAvg)", "—"]],
               col_w=[0.22,0.22,0.34,0.22], hdr_bg=C_TEAL)

    sub_band(fig, "1.8  Label Encoding Summary", 0.04, 0.735, 0.92)
    grid_table(fig, 0.04, 0.635, 0.92, 0.095,
               ["Column", "Original Values", "Encoded Values"],
               [["Variety",    "CO_43 · ponmani · delux ponni", "0 · 1 · 2"],
                ["Soil Types", "alluvial · clay",                "0 · 1"],
                ["Nursery",    "dry · wet",                      "0 · 1"]],
               col_w=[0.22,0.52,0.26], hdr_bg=C_TEAL)

    sub_band(fig, "1.9  One-Hot Encoded Wind Direction Columns (22 total)", 0.04, 0.590, 0.92)
    ohe_cols_list = [
        "Wind Direction_D1_D30_E", "Wind Direction_D1_D30_ENE", "Wind Direction_D1_D30_NW",
        "Wind Direction_D1_D30_SSE","Wind Direction_D1_D30_SW", "Wind Direction_D1_D30_W",
        "Wind Direction_D31_D60_ENE","Wind Direction_D31_D60_NE","Wind Direction_D31_D60_S",
        "Wind Direction_D31_D60_W", "Wind Direction_D31_D60_WNW",
        "Wind Direction_D61_D90_NE","Wind Direction_D61_D90_NNE","Wind Direction_D61_D90_NNW",
        "Wind Direction_D61_D90_SE","Wind Direction_D61_D90_SW",
        "Wind Direction_D91_D120_NNW","Wind Direction_D91_D120_NW","Wind Direction_D91_D120_S",
        "Wind Direction_D91_D120_SSE","Wind Direction_D91_D120_W","Wind Direction_D91_D120_WSW",
    ]
    # display as 2-column grid
    half = 11
    rows_ohe = [(ohe_cols_list[i], ohe_cols_list[i+half] if i+half<len(ohe_cols_list) else "")
                for i in range(half)]
    grid_table(fig, 0.04, 0.440, 0.92, 0.145,
               ["OHE Column (1–11)", "OHE Column (12–22)"],
               rows_ohe, col_w=[0.50,0.50], hdr_bg=C_TEAL, fs=7.2)

    # Feature count waterfall chart
    ax_wf = fig.add_axes([0.08, 0.08, 0.84, 0.33])
    ax_wf.set_facecolor(C_GRAY)
    stages = ["Raw\nDataset","After OHE\n+drop","After corr.\npruning","Final\nfeatures"]
    counts = [45, 61, 46, 45]
    colors_wf = [C_BLUE, C_ORANGE, C_RED, C_GREEN]
    bars = ax_wf.bar(stages, counts, color=colors_wf, width=0.5,
                     edgecolor=C_WHITE, lw=1.2)
    for bar, v in zip(bars, counts):
        ax_wf.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   str(v), ha="center", fontsize=10, fontweight="bold", color=C_DARK)
    ax_wf.set_title("Feature Count Through the Pipeline", fontsize=10,
                    fontweight="bold", color=C_NAVY, pad=8)
    ax_wf.set_ylabel("Number of Columns", fontsize=9)
    ax_wf.set_ylim(0, 75)
    ax_wf.tick_params(labelsize=9)
    ax_wf.spines["left"].set_color(C_LINE)
    ax_wf.spines["bottom"].set_color(C_LINE)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 5 — TASK 2 : Default Decision Tree
    # ──────────────────────────────────────────────────────────────────────────
    pg = 5
    fig = new_page(pdf)
    band(fig, "Task 2 — Decision Tree Classifier",
         "Default Model", bg=C_NAVY)

    sub_band(fig, "2.1  What is a Decision Tree?", 0.04, 0.875, 0.92)
    body_text(fig, [
        ("• A Decision Tree recursively partitions the feature space by choosing the split "
         "that maximises information gain (entropy) or minimises impurity (Gini)."),
        ("• Each internal node is a feature threshold test; leaves assign a class label. "
         "Depth and minimum-sample constraints control overfitting."),
        ("• Highly interpretable — the full decision path for any prediction can be "
         "traced from root to leaf."),
    ], 0.04, 0.775, 0.92, 0.095, fs=8.5, gap=1.9)

    sub_band(fig, "2.2  Default DecisionTreeClassifier (random_state=42)", 0.04, 0.730, 0.92)
    grid_table(fig, 0.04, 0.645, 0.92, 0.080,
               ["Metric", "Value"],
               [["Training Accuracy","93.90 %"],
                ["Test Accuracy",    "84.41 %"],
                ["Overfit Gap",      "9.49 pp (overfitted)"],
                ["Number of Nodes",  "545"],
                ["Tree Depth",       "16"]],
               col_w=[0.5,0.5], hdr_bg=C_NAVY)

    sub_band(fig, "2.3  Split Variables", 0.04, 0.600, 0.92)
    grid_table(fig, 0.04, 0.530, 0.92, 0.065,
               ["Split Level", "Variable"],
               [["1st split (root)","Trash(in bundles)"],
                ["2nd split — left branch","Wind Direction_D31_D60_W"],
                ["2nd split — right branch","Trash(in bundles)  (repeated)"]],
               col_w=[0.40,0.60], hdr_bg=C_NAVY)

    # Feature importance chart — default
    sub_band(fig, "2.4  Top-10 Feature Importances — Default Tree", 0.04, 0.485, 0.92)
    imp_def = pd.Series(dt_def.feature_importances_, index=feat).nlargest(10)
    ax_imp = fig.add_axes([0.08, 0.17, 0.84, 0.305])
    ax_imp.set_facecolor(C_GRAY)
    colors_imp = [C_ORANGE if i == 0 else C_BLUE for i in range(len(imp_def))]
    bars_imp = ax_imp.barh(imp_def.index[::-1], imp_def.values[::-1],
                           color=colors_imp[::-1], edgecolor=C_WHITE, lw=0.8)
    for bar, v in zip(bars_imp, imp_def.values[::-1]):
        ax_imp.text(v+0.002, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=7.5, color=C_DARK)
    ax_imp.set_xlabel("Importance Score", fontsize=8)
    ax_imp.tick_params(labelsize=7.5)
    ax_imp.set_xlim(0, imp_def.max()*1.18)
    ax_imp.spines["left"].set_color(C_LINE)
    ax_imp.spines["bottom"].set_color(C_LINE)

    # Code snippet
    sub_band(fig, "2.5  Key Code Snippet", 0.04, 0.132, 0.92)
    code_box(fig, [
        "from sklearn.tree import DecisionTreeClassifier",
        "dt_default = DecisionTreeClassifier(random_state=42)",
        "dt_default.fit(X_train, y_train)",
        "print('Train:', accuracy_score(y_train, dt_default.predict(X_train)))",
        "print('Test: ', accuracy_score(y_test,  dt_default.predict(X_test)))",
        "print('Nodes:', dt_default.tree_.node_count)",
        "print('Depth:', dt_default.get_depth())",
    ], 0.04, 0.03, 0.92, 0.098, fs=7.5)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 6 — TASK 2 : Tuned Decision Tree
    # ──────────────────────────────────────────────────────────────────────────
    pg = 6
    fig = new_page(pdf)
    band(fig, "Task 2 — Decision Tree Classifier",
         "Tuned Model (GridSearchCV)", bg=C_NAVY)

    sub_band(fig, "2.6  GridSearchCV Parameter Grid", 0.04, 0.875, 0.92)
    grid_table(fig, 0.04, 0.790, 0.92, 0.080,
               ["Hyperparameter", "Values Searched"],
               [["max_depth",         "[3, 5, 7, 10, None]"],
                ["min_samples_split", "[2, 5, 10, 20]"],
                ["min_samples_leaf",  "[1, 2, 5, 10]"],
                ["criterion",         "['gini', 'entropy']"]],
               col_w=[0.38,0.62], hdr_bg=C_NAVY)

    body_text(fig, [
        "cv = 5-fold cross-validation     ·     scoring = 'accuracy'     ·"
        "     Total combinations = 160"
    ], 0.04, 0.755, 0.92, 0.032, fs=8.2)

    sub_band(fig, "2.7  Best Parameters Found", 0.04, 0.720, 0.92)
    grid_table(fig, 0.04, 0.650, 0.92, 0.065,
               ["Parameter", "Best Value"],
               [["criterion",         "entropy"],
                ["max_depth",         "5"],
                ["min_samples_leaf",  "1"],
                ["min_samples_split", "2"]],
               col_w=[0.5,0.5], hdr_bg=C_NAVY)

    sub_band(fig, "2.8  Default vs Tuned — Performance Comparison", 0.04, 0.605, 0.92)
    grid_table(fig, 0.04, 0.510, 0.92, 0.090,
               ["Metric", "Default DT", "Tuned DT", "Improvement"],
               [["Training Accuracy", "93.90 %", "91.48 %", "−2.42 pp (less overfit)"],
                ["Test Accuracy",     "84.41 %", "89.61 %", "+5.20 pp ✓"],
                ["Overfit Gap",       "9.49 pp", "1.87 pp", "−7.62 pp ✓"],
                ["Number of Nodes",   "545",     "39",      "−506 nodes (93 % simpler)"],
                ["Tree Depth",        "16",      "5",       "−11 levels"]],
               col_w=[0.32,0.20,0.20,0.28], hdr_bg=C_NAVY)

    # Feature importance chart — tuned
    sub_band(fig, "2.9  Top-10 Feature Importances — Tuned Tree", 0.04, 0.465, 0.92)
    imp_tun = pd.Series(dt_tun.feature_importances_, index=feat).nlargest(10)
    ax_imp2 = fig.add_axes([0.08, 0.16, 0.84, 0.295])
    ax_imp2.set_facecolor(C_GRAY)
    cols2 = [C_ORANGE if i == 0 else C_BLUE for i in range(len(imp_tun))]
    bars2 = ax_imp2.barh(imp_tun.index[::-1], imp_tun.values[::-1],
                         color=cols2[::-1], edgecolor=C_WHITE, lw=0.8)
    for bar, v in zip(bars2, imp_tun.values[::-1]):
        ax_imp2.text(v+0.003, bar.get_y()+bar.get_height()/2,
                     f"{v:.4f}", va="center", fontsize=7.5, color=C_DARK)
    ax_imp2.set_xlabel("Importance Score", fontsize=8)
    ax_imp2.tick_params(labelsize=7.5)
    ax_imp2.set_xlim(0, imp_tun.max()*1.16)
    ax_imp2.spines["left"].set_color(C_LINE)
    ax_imp2.spines["bottom"].set_color(C_LINE)

    sub_band(fig, "2.10  Key Code Snippet — GridSearchCV", 0.04, 0.127, 0.92)
    code_box(fig, [
        "param_grid = {'max_depth':[3,5,7,10,None], 'min_samples_split':[2,5,10,20],",
        "              'min_samples_leaf':[1,2,5,10], 'criterion':['gini','entropy']}",
        "grid = GridSearchCV(DecisionTreeClassifier(random_state=42),",
        "                    param_grid, cv=5, scoring='accuracy', n_jobs=-1)",
        "grid.fit(X_train, y_train)",
        "dt_tuned = grid.best_estimator_",
        "print(grid.best_params_)   # {'criterion':'entropy','max_depth':5,...}",
    ], 0.04, 0.03, 0.92, 0.094, fs=7.5)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 7 — TASK 2 : Tree Visualisations
    # ──────────────────────────────────────────────────────────────────────────
    pg = 7
    fig = new_page(pdf)
    band(fig, "Task 2 — Decision Tree Classifier",
         "Tree Visualisations (max_depth = 3 view)", bg=C_NAVY)

    body_text(fig, [
        ("• Both trees are plotted to depth 3 for readability. Nodes are coloured by "
         "class majority: orange = BelowAvg, green = AboveAvg."),
        ("• The default tree root splits on Trash(in bundles) — the dominant agronomic "
         "signal in the dataset."),
    ], 0.04, 0.868, 0.92, 0.060, fs=8.5, gap=2.0)

    sub_band(fig, "Default Decision Tree (depth-3 view) — 545 nodes, depth 16", 0.04, 0.807, 0.92)
    img_ax(fig, 0.04, 0.47, 0.92, 0.328, MODELS/"tree_default.png")

    sub_band(fig, "Tuned Decision Tree (depth-3 view) — 39 nodes, depth 5", 0.04, 0.430, 0.92)
    img_ax(fig, 0.04, 0.10, 0.92, 0.325, MODELS/"tree_tuned.png")

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 8 — TASK 2 : ROC Curve
    # ──────────────────────────────────────────────────────────────────────────
    pg = 8
    fig = new_page(pdf)
    band(fig, "Task 2 — Decision Tree Classifier",
         "ROC Curves", bg=C_NAVY)

    body_text(fig, [
        ("• The ROC curve plots True Positive Rate vs False Positive Rate across all "
         "classification thresholds. AUC (Area Under Curve) closer to 1.0 is better."),
        ("• The tuned tree (AUC ≈ 0.90) substantially outperforms the default tree, "
         "confirming that pruning via GridSearchCV reduced overfitting."),
    ], 0.04, 0.868, 0.92, 0.065, fs=8.5, gap=2.0)

    sub_band(fig, "ROC Curves — Default vs Tuned Decision Tree", 0.04, 0.800, 0.92)
    img_ax(fig, 0.12, 0.35, 0.76, 0.44, MODELS/"roc_decision_trees.png")

    # Regenerate AUC values for inline table
    fpr_d, tpr_d, _ = roc_curve(y_test, dt_def.predict_proba(X_test)[:,1])
    fpr_t, tpr_t, _ = roc_curve(y_test, dt_tun.predict_proba(X_test)[:,1])
    auc_d = auc(fpr_d, tpr_d)
    auc_t = auc(fpr_t, tpr_t)

    sub_band(fig, "AUC Summary", 0.04, 0.297, 0.92)
    grid_table(fig, 0.04, 0.205, 0.92, 0.088,
               ["Model", "Test Accuracy", "AUC", "Verdict"],
               [["Default DT", "84.41 %", f"{auc_d:.4f}", "Overfitted — poor generalisation"],
                ["Tuned DT",   "89.61 %", f"{auc_t:.4f}", "Well-generalised — recommended ✓"]],
               col_w=[0.22,0.20,0.14,0.44], hdr_bg=C_NAVY)

    body_text(fig, [
        ("• A Decision Tree with max_depth=5 and criterion=entropy achieved 89.61 % test "
         "accuracy — a 5.2 percentage-point improvement over the unconstrained default."),
        ("• Trash(in bundles) alone captures 92 % of the information gain, suggesting "
         "harvest-side agronomic practices are the strongest yield predictor."),
    ], 0.04, 0.06, 0.92, 0.130, fs=8.5, gap=2.0)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 9 — TASK 3 : Full Logistic Regression
    # ──────────────────────────────────────────────────────────────────────────
    pg = 9
    fig = new_page(pdf)
    band(fig, "Task 3 — Logistic Regression",
         "Full Model (45 features)", bg=C_ORANGE)

    sub_band(fig, "3.1  What is Logistic Regression?", 0.04, 0.875, 0.92)
    body_text(fig, [
        ("• Logistic Regression models P(y=1|X) = σ(w·X + b) where σ is the sigmoid "
         "function. The decision boundary is a hyperplane in feature space."),
        ("• Regularisation parameter C controls penalty strength: small C = strong L2 "
         "regularisation (smaller coefficients, better generalisation)."),
        ("• Features must be standardised (zero mean, unit variance) so that coefficient "
         "magnitudes are comparable and gradient descent converges stably."),
    ], 0.04, 0.77, 0.92, 0.100, fs=8.5, gap=1.9)

    sub_band(fig, "3.2  GridSearchCV — Parameter Grid & Best Result", 0.04, 0.725, 0.92)
    grid_table(fig, 0.04, 0.650, 0.92, 0.070,
               ["Parameter", "Values Searched", "Best Value"],
               [["C",        "[0.01, 0.1, 1, 10, 100]",  "0.01"],
                ["solver",   "['lbfgs', 'liblinear']",    "liblinear"],
                ["max_iter", "[200, 500, 1000]",           "200"]],
               col_w=[0.25,0.45,0.30], hdr_bg=C_ORANGE)

    sub_band(fig, "3.3  Full Model Performance", 0.04, 0.605, 0.92)
    grid_table(fig, 0.04, 0.535, 0.92, 0.065,
               ["Metric", "Value"],
               [["Best CV Accuracy", "88.62 %"],
                ["Training Accuracy","89.02 %"],
                ["Test Accuracy",    "87.10 %"],
                ["Overfit Gap",      "0.92 pp  (very well generalised)"]],
               col_w=[0.45,0.55], hdr_bg=C_ORANGE)

    # Coefficient bar chart (full LR)
    sub_band(fig, "3.4  Top-10 Features by |Coefficient| — Full LR", 0.04, 0.490, 0.92)
    coefs_full = pd.Series(lr_full.coef_[0], index=feat)
    top10_full = coefs_full.abs().nlargest(10)
    ax_coef = fig.add_axes([0.08, 0.18, 0.84, 0.300])
    ax_coef.set_facecolor(C_GRAY)
    signed = coefs_full[top10_full.index][::-1]
    bar_colors = [C_GREEN if v > 0 else C_RED for v in signed.values]
    bars_c = ax_coef.barh(signed.index, signed.values,
                          color=bar_colors, edgecolor=C_WHITE, lw=0.7)
    ax_coef.axvline(0, color=C_DARK, lw=0.8, linestyle="--")
    for bar, v in zip(bars_c, signed.values):
        ha = "left" if v >= 0 else "right"
        offset = 0.006 if v >= 0 else -0.006
        ax_coef.text(v + offset, bar.get_y()+bar.get_height()/2,
                     f"{v:+.4f}", va="center", ha=ha, fontsize=7.5, color=C_DARK)
    ax_coef.set_xlabel("Coefficient Value (scaled features)", fontsize=8)
    ax_coef.tick_params(labelsize=7.5)
    ax_coef.spines["left"].set_color(C_LINE)
    ax_coef.spines["bottom"].set_color(C_LINE)

    # Legend
    from matplotlib.patches import Patch
    ax_coef.legend(handles=[Patch(color=C_GREEN, label="Positive → increases P(AboveAvg)"),
                             Patch(color=C_RED,   label="Negative → decreases P(AboveAvg)")],
                   fontsize=7.5, loc="lower right")

    sub_band(fig, "3.5  Key Code Snippet", 0.04, 0.147, 0.92)
    code_box(fig, [
        "scaler = StandardScaler()",
        "X_train_sc = scaler.fit_transform(X_train)   # fit on train only",
        "X_test_sc  = scaler.transform(X_test)        # apply same scale to test",
        "param_grid = {'C':[0.01,0.1,1,10,100],'solver':['lbfgs','liblinear'],'max_iter':[200,500,1000]}",
        "grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')",
        "grid.fit(X_train_sc, y_train)",
        "lr_full = grid.best_estimator_  # C=0.01, solver=liblinear",
    ], 0.04, 0.040, 0.92, 0.104, fs=7.2)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 10 — TASK 3 : Reduced Logistic Regression
    # ──────────────────────────────────────────────────────────────────────────
    pg = 10
    fig = new_page(pdf)
    band(fig, "Task 3 — Logistic Regression",
         "Reduced Model (top-10 DT features)", bg=C_ORANGE)

    sub_band(fig, "3.6  Why a Reduced Model?", 0.04, 0.875, 0.92)
    body_text(fig, [
        ("• The tuned Decision Tree assigned >99 % of its importance to just 10 features. "
         "We test whether Logistic Regression can match or beat the full model using only those."),
        ("• Fewer features → lower dimensionality, faster inference, easier to explain "
         "to domain experts, and less risk of noise amplification."),
    ], 0.04, 0.790, 0.92, 0.080, fs=8.5, gap=1.9)

    sub_band(fig, "3.7  Top-10 DT Features Used for Reduced Model", 0.04, 0.745, 0.92)
    grid_table(fig, 0.04, 0.630, 0.92, 0.110,
               ["#", "Feature", "DT Importance", "LR Coefficient"],
               [["1",  "Trash(in bundles)",               "0.9210", "+0.6917"],
                ["2",  "Seedrate(in Kg)",                 "0.0441", "+1.0941"],
                ["3",  "Variety",                         "0.0131", "−0.1598"],
                ["4",  "Soil Types",                      "0.0050", "−0.0029"],
                ["5",  "Relative Humidity_D1_D30",        "0.0046", "+0.0118"],
                ["6",  "Max temp_D31_D60",                "0.0036", "−0.0431"],
                ["7",  "Inst Wind Speed_D61_D90(in Knots)","0.0032","−0.0097"],
                ["8",  "Wind Direction_D1_D30_W",         "0.0017", "−0.0637"],
                ["9",  "Inst Wind Speed_D1_D30(in Knots)","0.0012", "+0.0370"],
                ["10", "Min temp_D31_D60",                "0.0010", "+0.0093"]],
               col_w=[0.06,0.44,0.25,0.25], hdr_bg=C_ORANGE)

    sub_band(fig, "3.8  Full vs Reduced — Performance Comparison", 0.04, 0.585, 0.92)
    grid_table(fig, 0.04, 0.495, 0.92, 0.085,
               ["Metric", "Full LR (45 feat.)", "Reduced LR (10 feat.)", "Winner"],
               [["Best CV Accuracy", "88.62 %", "89.02 %", "Reduced ✓"],
                ["Training Accuracy","89.02 %", "89.11 %", "Reduced ✓"],
                ["Test Accuracy",    "87.10 %", "87.46 %", "Reduced ✓"],
                ["Overfit Gap",      "0.92 pp", "1.65 pp", "Full (marginal)"],
                ["Features Used",    "45",      "10",      "Reduced (4.5× leaner) ✓"]],
               col_w=[0.28,0.22,0.25,0.25], hdr_bg=C_ORANGE)

    body_text(fig, [
        ("Key insight: Reducing to 10 features actually improved test accuracy by +0.36 pp. "
         "The 35 dropped features were contributing noise rather than signal."),
    ], 0.04, 0.447, 0.92, 0.038, fs=8.5)

    # Coefficient comparison chart: full vs reduced side by side
    sub_band(fig, "3.9  Coefficient Comparison — Full vs Reduced (shared features)", 0.04, 0.408, 0.92)
    shared = top10_dt
    coefs_r = pd.Series(lr_red.coef_[0], index=top10_dt)
    coefs_f_shared = pd.Series(lr_full.coef_[0], index=feat)[shared]
    x_pos = np.arange(len(shared))
    w_bar = 0.35
    ax_cc = fig.add_axes([0.06, 0.115, 0.88, 0.285])
    ax_cc.set_facecolor(C_GRAY)
    ax_cc.bar(x_pos - w_bar/2, coefs_f_shared.values, w_bar,
              label="Full LR",    color=C_BLUE,   edgecolor=C_WHITE, lw=0.6)
    ax_cc.bar(x_pos + w_bar/2, coefs_r.values,       w_bar,
              label="Reduced LR", color=C_ORANGE, edgecolor=C_WHITE, lw=0.6)
    ax_cc.axhline(0, color=C_DARK, lw=0.7)
    ax_cc.set_xticks(x_pos)
    short_names = [f.replace("Wind Direction_","WD_")
                    .replace("Inst Wind Speed_","IWS_")
                    .replace("Relative Humidity_","RH_")
                    .replace("(in Knots)","")
                    .replace("(in bundles)","") for f in shared]
    ax_cc.set_xticklabels(short_names, rotation=35, ha="right", fontsize=6.5)
    ax_cc.set_ylabel("Coefficient", fontsize=8)
    ax_cc.legend(fontsize=8, loc="upper right")
    ax_cc.tick_params(axis="y", labelsize=8)
    ax_cc.spines["left"].set_color(C_LINE)
    ax_cc.spines["bottom"].set_color(C_LINE)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 11 — TASK 3 : ROC Curve
    # ──────────────────────────────────────────────────────────────────────────
    pg = 11
    fig = new_page(pdf)
    band(fig, "Task 3 — Logistic Regression",
         "ROC Curves", bg=C_ORANGE)

    body_text(fig, [
        ("• Both Logistic Regression models produce smooth ROC curves (continuous probability "
         "scores), unlike Decision Trees which output probabilities only at leaf nodes."),
        ("• The reduced model's ROC curve closely tracks the full model — confirming that "
         "10 features capture essentially the same discriminative information as all 45."),
    ], 0.04, 0.868, 0.92, 0.068, fs=8.5, gap=2.0)

    sub_band(fig, "ROC Curves — Full vs Reduced Logistic Regression", 0.04, 0.800, 0.92)
    img_ax(fig, 0.12, 0.37, 0.76, 0.415, MODELS/"roc_logistic_regression.png")

    fpr_f, tpr_f, _ = roc_curve(y_test, lr_full.predict_proba(Xte_sc)[:,1])
    fpr_r, tpr_r, _ = roc_curve(y_test, lr_red.predict_proba(Xte_r)[:,1])
    auc_f = auc(fpr_f, tpr_f); auc_r = auc(fpr_r, tpr_r)

    sub_band(fig, "AUC Summary", 0.04, 0.328, 0.92)
    grid_table(fig, 0.04, 0.240, 0.92, 0.082,
               ["Model", "Features", "Test Accuracy", "AUC", "Verdict"],
               [["Full LR",    "45", "87.10 %", f"{auc_f:.4f}", "Strong baseline"],
                ["Reduced LR", "10", "87.46 %", f"{auc_r:.4f}", "Leaner & slightly better ✓"]],
               col_w=[0.18,0.12,0.18,0.14,0.38], hdr_bg=C_ORANGE)

    body_text(fig, [
        "## Key Takeaways from Task 3",
        "",
        ("• Logistic Regression with strong regularisation (C=0.01) generalises well — "
         "overfit gap under 2 pp for both models."),
        ("• Seedrate(in Kg) has the largest positive coefficient (+1.09), confirming the "
         "plantation density is strongly linked to above-average yield."),
        ("• Variety has a negative coefficient (−0.16): certain rice varieties are "
         "systematically associated with lower relative yield."),
        ("• The reduced model is preferred for deployment: same accuracy, far simpler, "
         "and easier to interpret for agronomists."),
    ], 0.04, 0.06, 0.92, 0.170, fs=8.5, gap=1.85)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 12 — OVERALL COMPARISON
    # ──────────────────────────────────────────────────────────────────────────
    pg = 12
    fig = new_page(pdf)
    band(fig, "Overall Model Comparison",
         "All Tasks Summary", bg=C_TEAL)

    sub_band(fig, "Summary Table — All Models", 0.04, 0.875, 0.92)
    grid_table(fig, 0.04, 0.758, 0.92, 0.112,
               ["Model", "Train Acc", "Test Acc", "AUC", "Nodes/Features", "Overfit Gap"],
               [["Default DT",    "93.90 %", "84.41 %", f"{auc_d:.4f}", "545 nodes",   "9.49 pp"],
                ["Tuned DT",      "91.48 %", "89.61 %", f"{auc_t:.4f}", "39 nodes",    "1.87 pp"],
                ["Full LR",       "89.02 %", "87.10 %", f"{auc_f:.4f}", "45 features", "0.92 pp"],
                ["Reduced LR",    "89.11 %", "87.46 %", f"{auc_r:.4f}", "10 features", "1.65 pp"]],
               col_w=[0.20,0.14,0.14,0.12,0.20,0.20], hdr_bg=C_TEAL)

    # Grouped bar chart: train vs test accuracy
    ax_cmp = fig.add_axes([0.06, 0.435, 0.55, 0.30])
    ax_cmp.set_facecolor(C_GRAY)
    models_lbl = ["Default DT", "Tuned DT", "Full LR", "Reduced LR"]
    train_accs = [93.90, 91.48, 89.02, 89.11]
    test_accs  = [84.41, 89.61, 87.10, 87.46]
    xp = np.arange(4)
    ax_cmp.bar(xp - 0.2, train_accs, 0.35, label="Train Accuracy",
               color=C_BLUE,   edgecolor=C_WHITE, lw=0.8)
    ax_cmp.bar(xp + 0.2, test_accs,  0.35, label="Test Accuracy",
               color=C_GREEN,  edgecolor=C_WHITE, lw=0.8)
    for i,(tr,te) in enumerate(zip(train_accs, test_accs)):
        ax_cmp.text(i-0.2, tr+0.3, f"{tr:.1f}", ha="center", fontsize=7, color=C_DARK)
        ax_cmp.text(i+0.2, te+0.3, f"{te:.1f}", ha="center", fontsize=7, color=C_DARK)
    ax_cmp.set_xticks(xp); ax_cmp.set_xticklabels(models_lbl, fontsize=8)
    ax_cmp.set_ylabel("Accuracy (%)", fontsize=8.5)
    ax_cmp.set_ylim(78, 98)
    ax_cmp.legend(fontsize=8); ax_cmp.tick_params(labelsize=8)
    ax_cmp.set_title("Train vs Test Accuracy", fontsize=9,
                     fontweight="bold", color=C_NAVY, pad=5)
    ax_cmp.spines["left"].set_color(C_LINE)
    ax_cmp.spines["bottom"].set_color(C_LINE)

    # AUC bar chart
    ax_auc = fig.add_axes([0.65, 0.435, 0.30, 0.30])
    ax_auc.set_facecolor(C_GRAY)
    aucs = [auc_d, auc_t, auc_f, auc_r]
    auc_colors = [C_RED, C_GREEN, C_BLUE, C_ORANGE]
    bars_auc = ax_auc.bar(models_lbl, aucs, color=auc_colors,
                          edgecolor=C_WHITE, lw=0.8)
    for bar, v in zip(bars_auc, aucs):
        ax_auc.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", fontsize=7.5, fontweight="bold", color=C_DARK)
    ax_auc.set_ylim(0.75, 1.0)
    ax_auc.set_xticklabels(models_lbl, fontsize=6.5, rotation=20, ha="right")
    ax_auc.set_ylabel("AUC", fontsize=8.5)
    ax_auc.set_title("AUC Comparison", fontsize=9,
                     fontweight="bold", color=C_NAVY, pad=5)
    ax_auc.tick_params(labelsize=8)
    ax_auc.spines["left"].set_color(C_LINE)
    ax_auc.spines["bottom"].set_color(C_LINE)

    sub_band(fig, "Conclusions & Recommendations", 0.04, 0.393, 0.92)
    body_text(fig, [
        ("## Best Overall Model: Tuned Decision Tree (89.61 % test accuracy, AUC "
         + f"{auc_t:.4f})"),
        "",
        ("• The tuned DT is the top performer across both accuracy and AUC. With only "
         "39 nodes and depth 5, it is interpretable and deployment-ready."),
        "",
        ("• Logistic Regression (both variants) generalises extremely well (overfit gap "
         "<2 pp) and makes a strong interpretable alternative. The reduced 10-feature "
         "version is recommended when a linear model is preferred."),
        "",
        ("• The single most important predictor across all models is Trash(in bundles) "
         "followed by Seedrate(in Kg) — suggesting post-harvest and planting practices "
         "dominate yield outcomes more than climatic factors."),
        "",
        ("• Future work: ensemble methods (Random Forest, Gradient Boosting), SMOTE "
         "for class balance experiments, and SHAP for model explainability."),
    ], 0.04, 0.08, 0.92, 0.305, fs=8.5, gap=1.5)

    footer(fig, pg, TOTAL_PAGES)
    save(pdf, fig)

print(f"\nPDF saved → {OUT_PDF}")
