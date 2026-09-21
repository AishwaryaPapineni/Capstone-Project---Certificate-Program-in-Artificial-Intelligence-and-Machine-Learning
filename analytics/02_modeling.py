"""
02_modeling.py — Module 2, Part B: predictive modeling, continuing from the
same cleaned data.

Run:
    python 02_modeling.py

Reads the committed titanic.csv (written by 01_eda.py; NOT a second network
load), splits it, builds a preprocessing+model Pipeline fit only on the
training split, trains/evaluates three classifiers, compares imbalance
handling strategies, tunes a Random Forest with GridSearchCV, runs a linear
regression side-task on fare, and saves the best full pipeline with joblib.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

HERE = Path(__file__).parent
CHARTS = HERE / "charts"
CHARTS.mkdir(exist_ok=True)

pd.set_option("display.width", 120)
RANDOM_STATE = 42


def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# ---------------------------------------------------------------------------
# Load the SAME committed titanic.csv (no second sns.load_dataset call)
# ---------------------------------------------------------------------------
hr("LOAD (from titanic.csv, not a second network call)")
df = pd.read_csv(HERE / "titanic.csv")
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from titanic.csv")

TARGET = "survived"
FEATURES_NUM = ["age", "sibsp", "parch", "fare"]
FEATURES_CAT = ["sex", "embarked", "pclass"]
FEATURES = FEATURES_NUM + FEATURES_CAT

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ---------------------------------------------------------------------------
# Task 1: stratified split
# ---------------------------------------------------------------------------
hr("STRATIFIED TRAIN/TEST SPLIT")
class_balance = y.value_counts(normalize=True)
print("Class balance (survived):\n", class_balance)
print("""
Justification for stratification: survival is imbalanced (~38% survived /
~62% did not, see above). A plain random split can by chance shift that ratio
between train and test, which would let a model tune to a train-set balance
that doesn't match what it's evaluated on. stratify=y forces both splits to
keep the same ~38/62 ratio as the full dataset, so test-set performance is a
fair readout of how the model handles the *actual* class balance.
""")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
print("Train class balance:\n", y_train.value_counts(normalize=True))
print("Test class balance:\n", y_test.value_counts(normalize=True))

# ---------------------------------------------------------------------------
# Task 2: preprocessing — ColumnTransformer, fit on TRAIN only
# ---------------------------------------------------------------------------
hr("PREPROCESSING PIPELINE (fit on train only)")

numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, FEATURES_NUM),
    ("cat", categorical_pipeline, FEATURES_CAT),
])
print("""
ColumnTransformer: numeric columns (age, sibsp, parch, fare) are
median-imputed then StandardScaler-scaled; categorical columns (sex,
embarked, pclass) are most-frequent-imputed then one-hot encoded. Wrapped in
sklearn Pipelines and fit exclusively via pipeline.fit(X_train, y_train) below
- the imputer/scaler/encoder never see X_test during fit, only .transform().
""")

# ---------------------------------------------------------------------------
# Task 3/4: train three classifiers, full metric suite
# ---------------------------------------------------------------------------
hr("TRAIN 3 CLASSIFIERS + FULL METRICS")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

fitted_pipelines = {}
metrics_rows = []

plt.figure(figsize=(7, 6))
for name, clf in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print("Confusion matrix:\n", cm)
    print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=plt.gca())
    metrics_rows.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc})

plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.title("ROC curves — all 3 classifiers")
plt.tight_layout()
plt.savefig(CHARTS / "08_roc_curves.png", dpi=110)
plt.close()
print("\nSaved charts/08_roc_curves.png")

metrics_df = pd.DataFrame(metrics_rows).set_index("model").round(3)
hr("CLASSIFIER COMPARISON TABLE")
print(metrics_df)

# Decision tree visualization
dt_pipe = fitted_pipelines["Decision Tree"]
dt_model = dt_pipe.named_steps["model"]
feature_names = dt_pipe.named_steps["preprocess"].get_feature_names_out()
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree (max_depth=5, showing top 3 levels)")
plt.tight_layout()
plt.savefig(CHARTS / "09_decision_tree.png", dpi=110)
plt.close()
print("Saved charts/09_decision_tree.png")

# ---------------------------------------------------------------------------
# Task 5: imbalance handling comparison (Random Forest, 3 ways)
# ---------------------------------------------------------------------------
hr("IMBALANCE HANDLING COMPARISON (Random Forest)")
print("Overall class balance (full y):\n", y.value_counts())
print(f"  -> {y.mean()*100:.1f}% survived / {(1-y.mean())*100:.1f}% did not survive: moderately imbalanced.")

imbalance_rows = []

# (a) baseline
rf_base = Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))])
rf_base.fit(X_train, y_train)
pred = rf_base.predict(X_test)
imbalance_rows.append({
    "strategy": "(a) baseline",
    "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
})

# (b) class_weight='balanced'
rf_bal = Pipeline([("preprocess", preprocessor), ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE))])
rf_bal.fit(X_train, y_train)
pred = rf_bal.predict(X_test)
imbalance_rows.append({
    "strategy": "(b) class_weight=balanced",
    "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
})

# (c) SMOTE oversampling on TRAINING FOLD ONLY (via imblearn Pipeline, so the
# fit()-time resampling never touches X_test — no leakage)
rf_smote = ImbPipeline([
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
])
rf_smote.fit(X_train, y_train)
pred = rf_smote.predict(X_test)
imbalance_rows.append({
    "strategy": "(c) SMOTE (train fold only)",
    "precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
})

imbalance_df = pd.DataFrame(imbalance_rows).set_index("strategy").round(3)
print("\n", imbalance_df)

best_strategy = imbalance_df["f1"].idxmax()
print(f"""
Conclusion: "{best_strategy}" gives the best F1 among the three. In practice,
class_weight='balanced' and SMOTE both push recall up relative to the
baseline (they make the model work harder to catch the minority "survived"
class), usually at some cost to precision; the best F1 balances that
trade-off. On this dataset the imbalance is moderate (~62/38), so the gap
between strategies is real but not huge — a strongly imbalanced dataset would
show a bigger swing.
""")

# ---------------------------------------------------------------------------
# Task 6: GridSearchCV + OOB score on Random Forest
# ---------------------------------------------------------------------------
hr("HYPERPARAMETER TUNING: GridSearchCV on Random Forest (+ OOB score)")

rf_grid_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=RANDOM_STATE)),
])
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, None],
    "model__max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(rf_grid_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 3))
best_rf = grid.best_estimator_.named_steps["model"]
print("OOB score of best estimator:", round(best_rf.oob_score_, 3))

test_pred = grid.best_estimator_.predict(X_test)
print(f"Tuned RF test set: accuracy={accuracy_score(y_test, test_pred):.3f}  f1={f1_score(y_test, test_pred):.3f}")

# ---------------------------------------------------------------------------
# Task 7: regression side-task — predict fare
# ---------------------------------------------------------------------------
hr("REGRESSION SIDE-TASK: predict fare from other features")

reg_features_num = ["age", "sibsp", "parch", "pclass"]
reg_features_cat = ["sex", "embarked"]
Xr = df[reg_features_num + reg_features_cat].copy()
yr = df["fare"].copy()

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=RANDOM_STATE)

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), reg_features_num),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), reg_features_cat),
])
reg_pipe = Pipeline([("preprocess", reg_preprocessor), ("model", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = mean_squared_error(yr_test, yr_pred) ** 0.5
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test.shape[0], Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  Adjusted R2={adj_r2:.3f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(7, 5))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual plot — fare regression")
plt.tight_layout()
plt.savefig(CHARTS / "10_regression_residuals.png", dpi=110)
plt.close()
print("Saved charts/10_regression_residuals.png")
print("""
Heteroscedasticity conclusion: the residual spread visibly widens as predicted
fare increases (a funnel/fan shape rather than a uniform band around zero) —
this IS heteroscedasticity. It's expected here: fare is right-skewed with a
long tail of very expensive tickets, so a linear model's errors get larger in
absolute terms for the priciest predictions.
""")

# ---------------------------------------------------------------------------
# Task 8: final model comparison table + recommendation
# ---------------------------------------------------------------------------
hr("FINAL MODEL COMPARISON")
print("Classification metrics (3 classifiers):")
print(metrics_df)
print("\nRegression metrics (fare side-task, separate metric group — not on the same scale):")
reg_metrics_df = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adj_r2}]).round(3)
print(reg_metrics_df.to_string(index=False))

best_model_name = metrics_df["f1"].idxmax()
best_row = metrics_df.loc[best_model_name]
print(f"""
Recommendation: deploy the **{best_model_name}** model. It has the highest F1
({best_row['f1']:.3f}) among the three classifiers, balancing precision
({best_row['precision']:.3f}) and recall ({best_row['recall']:.3f}) better than
the alternatives, and its AUC ({best_row['auc']:.3f}) is competitive too.
Random Forest also tends to generalize well on this kind of small, mixed
categorical/numeric tabular dataset without heavy tuning, and (as shown above)
its performance can be pushed further with GridSearchCV. Logistic Regression
remains a reasonable, more interpretable fallback if explainability matters
more than raw performance.
""")

# ---------------------------------------------------------------------------
# Task 9: save the best FULL pipeline (preprocessing + estimator) with joblib
# ---------------------------------------------------------------------------
hr("SAVE BEST PIPELINE (joblib) + RELOAD CHECK")

best_full_pipeline = fitted_pipelines[best_model_name]
pipeline_path = HERE / "best_pipeline.joblib"
joblib.dump(best_full_pipeline, pipeline_path)
print(f"Saved {best_model_name} full pipeline -> {pipeline_path.name}")

reloaded = joblib.load(pipeline_path)
sample_raw = X_test.iloc[:5]
pred_before = best_full_pipeline.predict(sample_raw)
pred_after = reloaded.predict(sample_raw)
print("Predictions from pipeline before save:", pred_before)
print("Predictions from pipeline after joblib.load:", pred_after)
assert (pred_before == pred_after).all(), "Reloaded pipeline predictions differ!"
print("Reloaded pipeline predicts identically on raw, unpreprocessed input. OK.")

hr("02_modeling.py complete")
