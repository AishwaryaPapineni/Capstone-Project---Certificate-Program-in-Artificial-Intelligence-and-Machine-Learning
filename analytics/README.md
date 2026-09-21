# Module 2 — Analytics Pipeline (`/analytics`)

One cohesive pipeline: load the Titanic dataset once, profile and clean it,
tell a visual data story about it, then build and rigorously evaluate a full
predictive-modeling pipeline on the same data.

## How to run

```bash
pip install -r requirements.txt

python 01_eda.py        # loads titanic (once), profiles, cleans, EDA story, saves titanic.csv
python 02_modeling.py   # reads titanic.csv, splits, trains/evaluates/tunes, saves best_pipeline.joblib
```

`sns.load_dataset("titanic")` is called exactly once, inside `01_eda.py`,
which immediately saves the raw result to `titanic.csv`. `02_modeling.py`
reads that same `titanic.csv` back with `pd.read_csv` — it never calls
`sns.load_dataset` again, so the whole module works offline after the first
run (the committed `titanic.csv` also lets grading work without any network
access at all).

All charts are saved as `.png` files in `charts/`. Full console output from
both scripts is captured in `eda_output.txt` and `modeling_output.txt`.

## Part A — Profiling, cleaning, and the data story

**Missing values** (measured on the raw 891-row, 15-column dataset):

| Column | % missing | Bracket | Strategy |
|---|---|---|---|
| `deck` | 77.22% | >30% (unreliable) | **Dropped the column** — with over 3 in 4 values missing, any imputed value is mostly guesswork, and an "unknown" category would just relabel ~77% of rows identically, adding little real signal to either the EDA or the model. |
| `age` | 19.87% | 5–30% | **Median-imputed** |
| `embarked` / `embark_town` | 0.22% | <5% | **Dropped those 2 rows** |

Shape after cleaning: **889 rows × 14 columns**.

**Univariate — age & fare** (see `charts/01_univariate_age_fare.png`):
- IQR outliers: **age → 65 outliers** (bounds [2.50, 54.50]); **fare → 114 outliers** (bounds [-26.76, 65.66]).
- `fare`: mean=32.10, median=14.45, mode=8.05 → **mean > median > mode, so fare is right-skewed**: a long tail of a small number of very expensive tickets pulls the mean well above the typical (median) fare.

**Bivariate — survival rate by boolean masking:**

| Breakdown | Survival rate |
|---|---|
| sex = male | 0.189 |
| sex = female | 0.740 |
| pclass = 1 | 0.626 |
| pclass = 2 | 0.473 |
| pclass = 3 | 0.242 |
| male, pclass 1 | 0.369 |
| male, pclass 2 | 0.157 |
| male, pclass 3 | 0.135 |
| female, pclass 1 | 0.967 |
| female, pclass 2 | 0.921 |
| female, pclass 3 | 0.500 |

**Correlation matrix** (6 numeric columns: `survived, pclass, age, sibsp,
parch, fare`; `adult_male`/`alone` excluded as derived/redundant flags) —
see `charts/02_correlation_heatmap.png`. Two strongest off-diagonal pairs by
|r|:
1. **`pclass` ↔ `fare`, r = −0.55** — `pclass` is coded 1=first, 2=second,
   3=third, so a *lower* pclass number is a *more expensive* cabin; lower
   pclass therefore goes with higher fare, giving a strong negative r.
2. **`sibsp` ↔ `parch`, r = 0.41** — both count family members travelling
   together (siblings/spouses vs. parents/children); passengers travelling
   with a larger family tend to score higher on both counts at once (e.g. a
   parent travelling with children usually also has a spouse aboard).

**Multivariate data story** (4 charts, `charts/03…06`):
1. **Survival by class & sex** — women survived at a far higher rate than
   men in every class, and the gap barely narrows even in third class: sex
   was the single strongest advantage, ahead of class. First-class women had
   the best odds of any group; third-class men, by far the worst.
2. **Age distribution by survival** — median age is similar for survivors
   and non-survivors, but survivors skew slightly younger with a denser
   cluster of children, consistent with "women and children first."
3. **Fare vs age, colored by survival** — survivors cluster at higher fares
   across all ages, while non-survivors dominate the low-fare band,
   reinforcing that fare (a wealth/class proxy) tracked survival more than
   age alone.
4. **Survival by embarkation town** — Cherbourg passengers survived at the
   highest rate (0.55), Southampton at the lowest (0.34); this is largely a
   proxy for the class mix boarding at each port (Cherbourg skewed more
   first-class) rather than the port itself having a causal effect.

**Standardization sanity check** (`charts/07_standardization_before_after.png`):
z-scoring `age` and `fare` on the full cleaned data moves both from
(mean=29.3, std=13.0) / (mean=32.1, std=49.7) to **mean ≈ 0, std ≈ 1** for
both — confirmed numerically in `eda_output.txt`. This check is exploratory
only; the modeling pipeline below does its own train-only scaling.

## Part B — Predictive modeling

**Stratified split**: survival is imbalanced (~38% survived / ~62% did not).
A plain random split can shift that ratio between train and test by chance;
`stratify=y` keeps both splits at the same ~38/62 ratio as the full dataset,
so test performance reflects the real class balance rather than a lucky or
unlucky split.

**Preprocessing** (`ColumnTransformer` inside a `Pipeline`, fit on the
training split only): numeric columns (`age, sibsp, parch, fare`) are
median-imputed then `StandardScaler`-scaled; categorical columns (`sex,
embarked, pclass`) are most-frequent-imputed then one-hot encoded. Because the
whole thing is one `Pipeline.fit(X_train, y_train)` call, none of the imputer/
scaler/encoder statistics are ever computed from the test split.

**Classifier comparison** (test set, from `modeling_output.txt`):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.843 |
| Decision Tree | 0.760 | 0.741 | 0.580 | 0.650 | 0.789 |
| Random Forest | 0.810 | 0.787 | 0.696 | 0.738 | 0.837 |

Confusion matrices, the decision tree plot (`charts/09_decision_tree.png`),
and ROC curves (`charts/08_roc_curves.png`) are in the run log / charts folder.

**Imbalance handling comparison** (Random Forest, 3 ways; SMOTE applied to
the training fold only, inside the pipeline, so it never leaks into the test
set):

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| (a) Baseline | 0.787 | 0.696 | 0.738 |
| (b) `class_weight='balanced'` | 0.790 | 0.710 | **0.748** |
| (c) SMOTE (train fold only) | 0.742 | 0.710 | 0.726 |

**Conclusion**: `class_weight='balanced'` gave the best F1 here. Both
`class_weight='balanced'` and SMOTE push recall above the baseline (they make
the model work harder on the minority "survived" class), usually at some
precision cost; on this moderately-imbalanced dataset (~62/38) the gap between
strategies is real but not dramatic — a more severely imbalanced dataset would
likely show a bigger swing between approaches.

**GridSearchCV tuning** (Random Forest; `n_estimators`, `max_depth`,
`max_features`, 5-fold CV on F1):
- Best params: `{'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 300}`
- Best CV F1: 0.743
- **OOB score of the tuned estimator: 0.819**

**Regression side-task** (predict `fare` from `age, sibsp, parch, pclass,
sex, embarked` with linear regression):
- MAE = 20.81, RMSE = 30.47, R² = 0.400, Adjusted R² = 0.379
- Residual plot: `charts/10_regression_residuals.png`
- **Heteroscedasticity conclusion**: the residual spread visibly widens as
  predicted fare increases (a funnel shape, not a uniform band around zero) —
  this **is heteroscedasticity**, expected because `fare` is right-skewed with
  a long tail of very expensive tickets, so a linear model's absolute errors
  grow for the priciest predictions.

**Final comparison — two separate metric groups (not directly comparable to each other):**

Classification:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.843 |
| Decision Tree | 0.760 | 0.741 | 0.580 | 0.650 | 0.789 |
| Random Forest | 0.810 | 0.787 | 0.696 | 0.738 | 0.837 |

Regression (fare):

| MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|
| 20.81 | 30.47 | 0.400 | 0.379 |

**Recommendation**: deploy the **Random Forest** classifier. It has the
highest F1 (0.738) among the three, balancing precision (0.787) and recall
(0.696) better than Logistic Regression or the Decision Tree, and a
competitive AUC (0.837). It also tunes further with GridSearchCV (OOB 0.819)
without much extra work. Logistic Regression remains a solid, more
interpretable fallback if explainability outweighs the last few points of F1.

**Saved pipeline**: `best_pipeline.joblib` is the complete fitted
preprocessing + Random Forest pipeline (`ColumnTransformer` + model as one
object), saved with `joblib.dump`. `02_modeling.py` reloads it with
`joblib.load` and confirms it predicts identically on raw, unpreprocessed
input — see the final section of `modeling_output.txt`.

## Files

- `01_eda.py`, `02_modeling.py` — the two ordered stages
- `titanic.csv` — raw offline fallback (saved immediately after the one
  `sns.load_dataset` call)
- `charts/` — all saved chart images
- `best_pipeline.joblib` — the saved full pipeline
- `eda_output.txt`, `modeling_output.txt` — full captured run logs
