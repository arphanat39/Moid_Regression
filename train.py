# ============================================================
# ASTEROID MOID REGRESSION
# ============================================================
#
# Predict Minimum Orbit Intersection Distance (MOID)
# using physical and orbital characteristics.
#
# Target:
#   moid
#
# Features:
#   H
#   albedo
#   e
#   a
#   i
#   om
#   w
#   ma
#
# ============================================================


# ============================================================
# 1. MATPLOTLIB BACKEND
# ============================================================

# IMPORTANT:
# Must be before importing matplotlib.pyplot
# Prevents Windows Tkinter errors.

import matplotlib

matplotlib.use("Agg")


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


warnings.filterwarnings("ignore")


# ============================================================
# 3. SETTINGS
# ============================================================

DATA_PATH = "data/asteroid.csv"

MODEL_DIR = "model"
OUTPUT_DIR = "outputs"

TARGET = "moid"

FEATURES = [
    "H",
    "albedo",
    "e",
    "a",
    "i",
    "om",
    "w",
    "ma"
]

TEST_SIZE = 0.20

RANDOM_STATE = 42

# Maximum number of rows used for tree-model training.
#
# This prevents Random Forest / HistGradientBoosting
# from taking an extremely long time on a huge dataset.
#
# If your dataset has fewer rows, all rows are used.
MAX_TREE_TRAIN_ROWS = 150_000


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("=" * 70)
print("ASTEROID MOID REGRESSION")
print("=" * 70)

print("\n[1/9] Loading dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure your CSV is located at:\n"
        "data/asteroid.csv"
    )


df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2/9] Checking columns...")

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )


print("All required columns found.")


# ============================================================
# 7. CONVERT FEATURES TO NUMERIC
# ============================================================

print("\n[3/9] Cleaning data...")

for column in required_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove invalid target values
df = df[
    df[TARGET].notna()
]


# MOID should not be negative
df = df[
    df[TARGET] >= 0
]


# Remove duplicate rows
before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

duplicates_removed = (
    before_duplicates -
    after_duplicates
)


print(
    f"Duplicates removed: "
    f"{duplicates_removed:,}"
)


# ============================================================
# 8. REMOVE ROWS WITH ALL FEATURES MISSING
# ============================================================

df = df.dropna(
    subset=FEATURES,
    how="all"
)


print(
    f"Rows after cleaning: "
    f"{len(df):,}"
)


# ============================================================
# 9. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Final dataset shape: {df.shape}")

print("\nMissing values:")

missing_summary = (
    df[required_columns]
    .isna()
    .sum()
)

print(missing_summary)


# ============================================================
# 10. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive statistics:")

print(
    df[required_columns]
    .describe()
)


# Save statistics
df[required_columns].describe().to_csv(
    os.path.join(
        OUTPUT_DIR,
        "descriptive_statistics.csv"
    )
)


# ============================================================
# 11. EDA - TARGET DISTRIBUTION
# ============================================================

print("\n[4/9] Creating EDA plots...")


plt.figure(figsize=(10, 6))

sns.histplot(
    df[TARGET],
    bins=100,
    kde=True
)

plt.title(
    "Distribution of Minimum Orbit Intersection Distance (MOID)"
)

plt.xlabel(
    "MOID (AU)"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "moid_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 12. EDA - FEATURE DISTRIBUTIONS
# ============================================================

for feature in FEATURES:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        df[feature].dropna(),
        bins=80
    )

    plt.title(
        f"Distribution of {feature}"
    )

    plt.xlabel(feature)

    plt.ylabel("Frequency")

    plt.tight_layout()

    safe_name = feature.replace(
        " ",
        "_"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_distribution.png"
        ),
        dpi=120
    )

    plt.close()


# ============================================================
# 13. CORRELATION MATRIX
# ============================================================

plt.figure(
    figsize=(11, 8)
)

correlation = df[
    FEATURES + [TARGET]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_matrix.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

print("\n[5/9] Creating train/test split...")

X = df[FEATURES]

y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


print(
    f"Full training set: "
    f"{len(X_train):,} rows"
)

print(
    f"Test set: "
    f"{len(X_test):,} rows"
)


# ============================================================
# 15. LIMIT TREE TRAINING DATA
# ============================================================
#
# If the asteroid dataset is enormous, training Random Forest
# and HistGradientBoosting on every row can take a very long
# time.
#
# We keep the test set untouched.
#
# Only the training data used by tree models is limited.
#
# Linear Regression still uses the complete training set.
# ============================================================

if len(X_train) > MAX_TREE_TRAIN_ROWS:

    print(
        "\nLarge dataset detected."
    )

    print(
        f"Tree models will use "
        f"{MAX_TREE_TRAIN_ROWS:,} "
        f"random training rows."
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    selected_indices = rng.choice(
        len(X_train),
        size=MAX_TREE_TRAIN_ROWS,
        replace=False
    )

    X_tree = X_train.iloc[
        selected_indices
    ].copy()

    y_tree = y_train.iloc[
        selected_indices
    ].copy()

else:

    X_tree = X_train.copy()

    y_tree = y_train.copy()


print(
    f"Tree-model training rows: "
    f"{len(X_tree):,}"
)


# ============================================================
# 16. PREPROCESSING
# ============================================================

print("\n[6/9] Preparing models...")


# Linear Regression
#
# Median imputation:
#   fills missing values using training-set medians.
#
# StandardScaler:
#   scales features for Linear Regression.

linear_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)


# Random Forest
#
# Tree models don't require scaling.

random_forest_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=50,
                max_depth=12,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# HistGradientBoosting
#
# Usually much faster than standard GradientBoosting
# on large datasets.

hist_gradient_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ============================================================
# 17. MODEL TRAINING
# ============================================================

models = {
    "Linear Regression": linear_pipeline,

    "Random Forest": random_forest_pipeline,

    "HistGradientBoosting": hist_gradient_pipeline
}


results = []

trained_models = {}


# ------------------------------------------------------------
# LINEAR REGRESSION
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Training Linear Regression...")
print("-" * 70)

linear_pipeline.fit(
    X_train,
    y_train
)

print("Linear Regression finished.")

linear_predictions = (
    linear_pipeline.predict(
        X_test
    )
)


# Metrics
linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


results.append(
    {
        "Model": "Linear Regression",
        "MAE": linear_mae,
        "RMSE": linear_rmse,
        "R2": linear_r2
    }
)

trained_models[
    "Linear Regression"
] = linear_pipeline


print(
    f"MAE  : {linear_mae:.6f}"
)

print(
    f"RMSE : {linear_rmse:.6f}"
)

print(
    f"R²   : {linear_r2:.6f}"
)


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Training Random Forest...")
print("-" * 70)

print(
    "50 trees | max depth = 12"
)

random_forest_pipeline.fit(
    X_tree,
    y_tree
)

print("Random Forest finished.")

rf_predictions = (
    random_forest_pipeline.predict(
        X_test
    )
)


rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


results.append(
    {
        "Model": "Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "R2": rf_r2
    }
)

trained_models[
    "Random Forest"
] = random_forest_pipeline


print(
    f"MAE  : {rf_mae:.6f}"
)

print(
    f"RMSE : {rf_rmse:.6f}"
)

print(
    f"R²   : {rf_r2:.6f}"
)


# ------------------------------------------------------------
# HISTGRADIENTBOOSTING
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("Training HistGradientBoosting...")
print("-" * 70)

print(
    "150 iterations | 31 leaves"
)

hist_gradient_pipeline.fit(
    X_tree,
    y_tree
)

print(
    "HistGradientBoosting finished."
)

hist_predictions = (
    hist_gradient_pipeline.predict(
        X_test
    )
)


hist_mae = mean_absolute_error(
    y_test,
    hist_predictions
)

hist_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        hist_predictions
    )
)

hist_r2 = r2_score(
    y_test,
    hist_predictions
)


results.append(
    {
        "Model": "HistGradientBoosting",
        "MAE": hist_mae,
        "RMSE": hist_rmse,
        "R2": hist_r2
    }
)

trained_models[
    "HistGradientBoosting"
] = hist_gradient_pipeline


print(
    f"MAE  : {hist_mae:.6f}"
)

print(
    f"RMSE : {hist_rmse:.6f}"
)

print(
    f"R²   : {hist_r2:.6f}"
)


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

print("\n[7/9] Comparing models...")

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 19. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)

print(
    f"BEST MODEL: {best_model_name}"
)

print("=" * 70)


# ============================================================
# 20. GET BEST MODEL PREDICTIONS
# ============================================================

best_predictions = (
    best_model.predict(
        X_test
    )
)


# ============================================================
# 21. PREDICTION ERROR ANALYSIS
# ============================================================

print("\n[8/9] Creating prediction error analysis...")


error_df = pd.DataFrame(
    {
        "Actual_MOID_AU":
            y_test.values,

        "Predicted_MOID_AU":
            best_predictions
    }
)


error_df["Error_AU"] = (
    error_df["Predicted_MOID_AU"]
    -
    error_df["Actual_MOID_AU"]
)


error_df["Absolute_Error_AU"] = (
    error_df["Error_AU"]
    .abs()
)


error_df = error_df.sort_values(
    by="Absolute_Error_AU",
    ascending=False
)


error_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "prediction_errors.csv"
    ),
    index=False
)


# ============================================================
# 22. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.4,
    s=10
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)


plt.xlabel(
    "Actual MOID (AU)"
)

plt.ylabel(
    "Predicted MOID (AU)"
)

plt.title(
    f"Actual vs Predicted MOID\n"
    f"{best_model_name}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 23. RESIDUAL PLOT
# ============================================================

residuals = (
    y_test.values
    -
    best_predictions
)


plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    best_predictions,
    residuals,
    alpha=0.4,
    s=10
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted MOID (AU)"
)

plt.ylabel(
    "Residual (AU)"
)

plt.title(
    "Residual Analysis"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "residual_plot.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 24. FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating feature importance..."
)


feature_importance_df = None


# Random Forest
if best_model_name == "Random Forest":

    rf_model = (
        best_model.named_steps["model"]
    )

    feature_importance = (
        rf_model.feature_importances_
    )

    feature_importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,

            "Importance":
                feature_importance
        }
    )


# HistGradientBoosting
#
# HistGradientBoosting does not expose the same
# feature_importances_ attribute as Random Forest.
#
# We therefore use permutation importance.

elif best_model_name == "HistGradientBoosting":

    from sklearn.inspection import permutation_importance

    print(
        "Calculating permutation feature importance..."
    )

    # To avoid making this step slow on an enormous test set,
    # use at most 20,000 test rows.

    if len(X_test) > 20_000:

        rng = np.random.RandomState(
            RANDOM_STATE
        )

        indices = rng.choice(
            len(X_test),
            size=20_000,
            replace=False
        )

        X_importance = X_test.iloc[
            indices
        ]

        y_importance = y_test.iloc[
            indices
        ]

    else:

        X_importance = X_test

        y_importance = y_test


    permutation = permutation_importance(
        best_model,
        X_importance,
        y_importance,
        n_repeats=3,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )


    feature_importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,

            "Importance":
                permutation.importances_mean
        }
    )


# Linear Regression
elif best_model_name == "Linear Regression":

    linear_model = (
        best_model.named_steps["model"]
    )

    coefficients = np.abs(
        linear_model.coef_
    )

    feature_importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,

            "Importance": coefficients
        }
    )


# ============================================================
# 25. SAVE FEATURE IMPORTANCE
# ============================================================

if feature_importance_df is not None:

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    feature_importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.csv"
        ),
        index=False
    )


    plt.figure(
        figsize=(9, 6)
    )

    sns.barplot(
        data=feature_importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"Feature Importance\n"
        f"{best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 26. SAVE MODEL
# ============================================================

print("\n[9/9] Saving model...")


MODEL_SAVE_PATH = os.path.join(
    MODEL_DIR,
    "moid_model.joblib"
)


joblib.dump(
    best_model,
    MODEL_SAVE_PATH
)


# ============================================================
# 27. SAVE PROJECT INFORMATION
# ============================================================

project_info = {

    "target":
        TARGET,

    "target_unit":
        "AU",

    "features":
        FEATURES,

    "test_size":
        TEST_SIZE,

    "random_state":
        RANDOM_STATE,

    "best_model":
        best_model_name,

    "model_results":
        results_df.to_dict(
            orient="records"
        ),

    "training_rows_tree_models":
        len(X_tree),

    "full_training_rows":
        len(X_train),

    "test_rows":
        len(X_test)
}


joblib.dump(
    project_info,

    os.path.join(
        MODEL_DIR,
        "project_info.joblib"
    )
)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Dataset rows: "
    f"{len(df):,}"
)

print(
    f"Training rows: "
    f"{len(X_train):,}"
)

print(
    f"Tree training rows: "
    f"{len(X_tree):,}"
)

print(
    f"Test rows: "
    f"{len(X_test):,}"
)

print(
    f"\nBest model: "
    f"{best_model_name}"
)


best_result = results_df[
    results_df["Model"] ==
    best_model_name
].iloc[0]


print(
    f"MAE:  "
    f"{best_result['MAE']:.6f} AU"
)

print(
    f"RMSE: "
    f"{best_result['RMSE']:.6f} AU"
)

print(
    f"R²:   "
    f"{best_result['R2']:.6f}"
)


print(
    f"\nModel saved to:"
)

print(
    MODEL_SAVE_PATH
)


print(
    "\nOutput files saved to:"
)

print(
    OUTPUT_DIR
)


print("\nFiles generated:")

print(
    "  - model/moid_model.joblib"
)

print(
    "  - model/project_info.joblib"
)

print(
    "  - outputs/model_comparison.csv"
)

print(
    "  - outputs/prediction_errors.csv"
)

print(
    "  - outputs/feature_importance.csv"
)

print(
    "  - outputs/feature_importance.png"
)

print(
    "  - outputs/actual_vs_predicted.png"
)

print(
    "  - outputs/residual_plot.png"
)

print(
    "  - outputs/correlation_matrix.png"
)

print(
    "  - outputs/moid_distribution.png"
)

print(
    "  - outputs/descriptive_statistics.csv"
)

print("=" * 70)
print("DONE")
print("=" * 70)