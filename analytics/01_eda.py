"""
01_eda.py — Module 2, Part A: profiling, cleaning, and the data story.

Run:
    python 01_eda.py

Loads the Titanic dataset via seaborn's built-in loader (network/cache, exactly
ONCE for the whole module), immediately saves it as titanic.csv (the committed
offline fallback used by every later step, including 02_modeling.py), profiles
it, cleans it using a percentage-based missing-value threshold rule, and
produces the full univariate / bivariate / multivariate "data story" plus an
exploratory standardization sanity check.

All charts are saved as .png files into charts/. All printed output is also
captured to eda_output.txt (redirect stdout, e.g. `python 01_eda.py > eda_output.txt`)
for the required textual record.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# ---------------------------------------------------------------------------
# Load ONCE, save the committed offline fallback immediately
# ---------------------------------------------------------------------------
hr("LOAD")
df_raw = sns.load_dataset("titanic")
df_raw.to_csv(HERE / "titanic.csv", index=False)
print(f"Loaded {df_raw.shape[0]} rows x {df_raw.shape[1]} columns from seaborn (network/cache, one time only).")
print(f"Saved raw fallback -> titanic.csv (this is what 02_modeling.py reads back, no second network load).")

# ---------------------------------------------------------------------------
# Task 1: profiling
# ---------------------------------------------------------------------------
hr("PROFILING: df.info()")
df_raw.info()

hr("PROFILING: df.describe()")
print(df_raw.describe(include="all").transpose())

hr("PROFILING: df.shape")
print(df_raw.shape)

hr("PROFILING: % missing per column (columns with any missing values)")
missing_pct = (df_raw.isna().mean() * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print(missing_pct)

# ---------------------------------------------------------------------------
# Task 2: missing-value handling, threshold rule
#   <5%    -> drop rows
#   5-30%  -> impute
#   >30%   -> too unreliable to impute -> drop column or encode "missing" category
# ---------------------------------------------------------------------------
hr("CLEANING")
df = df_raw.copy()

print("Measured missing-value percentages (affected columns only):")
for col, pct in missing_pct.items():
    print(f"  {col:15s} {pct:6.2f}%")

# age: ~19.87% missing -> 5-30% bracket -> impute (median, robust to the fare/age
# right-skew we confirm below)
age_pct = missing_pct.get("age", 0)
if age_pct:
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)
    print(f"\nage: {age_pct:.2f}% missing (5-30% bracket) -> imputed with median age = {median_age:.1f}")

# embarked / embark_town: ~0.22% missing -> <5% bracket -> drop those rows
for col in ("embarked", "embark_town"):
    pct = missing_pct.get(col, 0)
    if pct and 0 < pct < 5:
        before = len(df)
        df = df[df[col].notna()]
        print(f"{col}: {pct:.2f}% missing (<5% bracket) -> dropped {before - len(df)} row(s)")

# deck: ~77.2% missing -> so high that imputation would be unreliable.
# Decision: DROP the column. Justification: with over 3 in 4 values missing,
# any imputed "deck" would be almost entirely guesswork rather than signal, and
# an "unknown deck" category would just relabel ~77% of rows identically,
# adding a near-constant column that carries little real information for either
# the EDA story or the downstream model. (We do NOT use this column in Part B
# either, for the same reason.)
deck_pct = missing_pct.get("deck", 0)
if deck_pct:
    df = df.drop(columns=["deck"])
    print(f"deck: {deck_pct:.2f}% missing (>30%, too unreliable to impute) -> DROPPED the column")

print(f"\nShape after cleaning: {df.shape}")
print("Remaining missing values:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------------------
# Task 3: Univariate analysis — age & fare
# ---------------------------------------------------------------------------
hr("UNIVARIATE: age & fare")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df["age"], bins=30, kde=True, ax=axes[0, 0]).set_title("Age — histogram")
sns.boxplot(x=df["age"], ax=axes[0, 1]).set_title("Age — box plot")
sns.histplot(df["fare"], bins=30, kde=True, ax=axes[1, 0]).set_title("Fare — histogram")
sns.boxplot(x=df["fare"], ax=axes[1, 1]).set_title("Fare — box plot")
plt.tight_layout()
plt.savefig(CHARTS / "01_univariate_age_fare.png", dpi=110)
plt.close()
print("Saved charts/01_univariate_age_fare.png")


def iqr_outliers(series: pd.Series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    return mask.sum(), lo, hi


for col in ("age", "fare"):
    n_out, lo, hi = iqr_outliers(df[col])
    print(f"{col}: IQR outlier bounds = [{lo:.2f}, {hi:.2f}]  ->  {n_out} outlier(s) out of {len(df)} rows")

fare_mean, fare_median, fare_mode = df["fare"].mean(), df["fare"].median(), df["fare"].mode().iloc[0]
print(f"\nfare: mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
if fare_mean > fare_median > fare_mode:
    skew_txt = "right-skewed (mean > median > mode) — a long tail of a few very expensive fares pulls the mean up."
elif fare_mean < fare_median < fare_mode:
    skew_txt = "left-skewed (mean < median < mode)."
else:
    skew_txt = "roughly symmetric (mean, median and mode are close together)."
print(f"Skewness conclusion: fare is {skew_txt}")

# ---------------------------------------------------------------------------
# Task 4: Bivariate analysis
# ---------------------------------------------------------------------------
hr("BIVARIATE: survival rate breakdowns (boolean masking)")

rate_by_sex = {}
for s in df["sex"].unique():
    mask = df["sex"] == s
    rate_by_sex[s] = df.loc[mask, "survived"].mean()
print("Survival rate by sex:")
for k, v in rate_by_sex.items():
    print(f"  {k:8s} {v:.3f}")

rate_by_pclass = {}
for p in sorted(df["pclass"].unique()):
    mask = df["pclass"] == p
    rate_by_pclass[p] = df.loc[mask, "survived"].mean()
print("\nSurvival rate by pclass:")
for k, v in rate_by_pclass.items():
    print(f"  class {k}  {v:.3f}")

print("\nSurvival rate by sex AND pclass (combined boolean mask):")
rows = []
for s in df["sex"].unique():
    for p in sorted(df["pclass"].unique()):
        mask = (df["sex"] == s) & (df["pclass"] == p)
        rate = df.loc[mask, "survived"].mean()
        rows.append((s, p, rate))
        print(f"  sex={s:8s} pclass={p}  survival_rate={rate:.3f}")

hr("BIVARIATE: correlation matrix (6 numeric columns) + heatmap")
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[corr_cols].corr()
print(corr.round(3))

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix (6 numeric columns)")
plt.tight_layout()
plt.savefig(CHARTS / "02_correlation_heatmap.png", dpi=110)
plt.close()
print("Saved charts/02_correlation_heatmap.png")

# rank off-diagonal pairs by |correlation|
pairs = []
for i, c1 in enumerate(corr_cols):
    for c2 in corr_cols[i + 1:]:
        pairs.append((c1, c2, abs(corr.loc[c1, c2]), corr.loc[c1, c2]))
pairs.sort(key=lambda x: x[2], reverse=True)
print("\nTop 2 strongest off-diagonal correlations:")
for c1, c2, absval, signed in pairs[:2]:
    print(f"  {c1} <-> {c2}: r = {signed:.3f}")

print(f"""
Interpretation: the two strongest relationships are
({pairs[0][0]} <-> {pairs[0][1]}, r={pairs[0][3]:.2f}) and
({pairs[1][0]} <-> {pairs[1][1]}, r={pairs[1][3]:.2f}).
`pclass` and `fare` are strongly negatively correlated because pclass is coded
1=first, 2=second, 3=third — i.e. a *lower* pclass number is a *more expensive*
cabin — so a lower pclass number goes with a higher fare, giving a strong
negative r. `sibsp` and `parch` are positively correlated because both count
family members traveling together (siblings/spouses vs. parents/children):
passengers travelling with a larger family tend to show up higher on both
counts at once, e.g. a parent travelling with several children has high parch
and is also likely to have a spouse aboard (sibsp>0).
""")

# ---------------------------------------------------------------------------
# Task 5: multivariate "data story" — at least 4 charts, each interpreted
# ---------------------------------------------------------------------------
hr("MULTIVARIATE DATA STORY")

# Chart 1: survival rate by sex and pclass (grouped bar)
plt.figure(figsize=(7, 5))
story1 = df.groupby(["pclass", "sex"])["survived"].mean().reset_index()
sns.barplot(data=story1, x="pclass", y="survived", hue="sex")
plt.title("Survival rate by class and sex")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig(CHARTS / "03_story_survival_by_class_sex.png", dpi=110)
plt.close()
print("""
Chart 1 (survival by class & sex): Women survived at a far higher rate than
men in every class, and the gap barely narrows even in third class — being
female was the single strongest advantage in this disaster, well ahead of
class. First-class women survived at the highest rate of any group; third-
class men at the lowest, by a wide margin.
""")

# Chart 2: age distribution by survival (box)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="survived", y="age")
plt.title("Age distribution by survival outcome")
plt.xlabel("Survived (0=No, 1=Yes)")
plt.tight_layout()
plt.savefig(CHARTS / "04_story_age_by_survival.png", dpi=110)
plt.close()
print("""
Chart 2 (age by survival): Median age is similar for survivors and
non-survivors, but survivors skew slightly younger with a visibly denser
cluster of children/young passengers — consistent with "women and children
first" loading procedures rather than age alone driving the outcome.
""")

# Chart 3: fare vs age scatter, colored by survival
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.6)
plt.title("Fare vs age, colored by survival")
plt.tight_layout()
plt.savefig(CHARTS / "05_story_fare_age_survival.png", dpi=110)
plt.close()
print("""
Chart 3 (fare vs age, by survival): Survivors (orange) are noticeably denser
at higher fares across all ages, while non-survivors (blue) dominate the
low-fare band. This reinforces that fare (a proxy for class/wealth) tracked
survival more than age did on its own.
""")

# Chart 4: survival rate by embarkation town
plt.figure(figsize=(7, 5))
story4 = df.groupby("embark_town")["survived"].mean().reset_index().sort_values("survived", ascending=False)
sns.barplot(data=story4, x="embark_town", y="survived")
plt.title("Survival rate by embarkation town")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig(CHARTS / "06_story_survival_by_embark_town.png", dpi=110)
plt.close()
print(f"""
Chart 4 (survival by embarkation town): Passengers who boarded at
{story4.iloc[0]['embark_town']} survived at the highest rate
({story4.iloc[0]['survived']:.2f}), those from {story4.iloc[-1]['embark_town']}
at the lowest ({story4.iloc[-1]['survived']:.2f}). This is largely a proxy for
class mix at each port (Cherbourg embarked a wealthier, more first-class-heavy
group) rather than embarkation port having any causal effect of its own.
""")

print("""
Overall story: survival on the Titanic was driven mainly by sex (women far
more likely to survive than men in every class), then by class/fare (wealth
bought a better chance via cabin location and earlier access to lifeboats),
with age playing a smaller, secondary role mostly visible among children.
""")

# ---------------------------------------------------------------------------
# Task 6 (EDA-stage only): z-score standardization sanity check
# ---------------------------------------------------------------------------
hr("EXPLORATORY STANDARDIZATION CHECK (age & fare) — EDA sanity check only, NOT used by the modeling pipeline")

check = df[["age", "fare"]].copy()
for col in ("age", "fare"):
    mean, std = check[col].mean(), check[col].std()
    check[f"{col}_z"] = (check[col] - mean) / std

print("Before standardization:")
print(check[["age", "fare"]].agg(["mean", "std"]).round(3))
print("\nAfter standardization (z = (x - mean) / std):")
print(check[["age_z", "fare_z"]].agg(["mean", "std"]).round(3))
print("\n-> both z-columns have mean ~0 and std ~1, confirming the transform is correct.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(check["age"], kde=True, ax=axes[0], color="steelblue").set_title("age (raw)")
sns.histplot(check["age_z"], kde=True, ax=axes[0], color="orange", alpha=0.5)
axes[0].legend(["raw", "z-scored"])
sns.histplot(check["fare"], kde=True, ax=axes[1], color="steelblue").set_title("fare (raw)")
sns.histplot(check["fare_z"], kde=True, ax=axes[1], color="orange", alpha=0.5)
axes[1].legend(["raw", "z-scored"])
plt.tight_layout()
plt.savefig(CHARTS / "07_standardization_before_after.png", dpi=110)
plt.close()
print("Saved charts/07_standardization_before_after.png")

hr("01_eda.py complete")
print("Cleaned in-memory df shape:", df.shape)
print("(titanic.csv on disk is the RAW fallback per the assignment spec; 02_modeling.py")
print(" reads titanic.csv and does its own train-only preprocessing, as required.)")
