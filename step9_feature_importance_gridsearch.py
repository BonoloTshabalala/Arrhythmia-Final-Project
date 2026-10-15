"""
STEP 8 — Feature Importance, GridSearchCV + SMOTE pipeline model
======================================================================
Identical logic to step8_feature_importance_relaxed.py, pointed at the
tuned pipeline model from step6_random_forest_gridsearch.py instead.
====================================================================
Goal: identify which of the 9 features actually drive the model's
predictions, and check this directly against H1's specific biomarker
claim - that RR-interval and morphological features (QRS duration,
P-wave amplitude) would emerge as the dominant predictors.

Two DIFFERENT importance methods are used deliberately, because they
can disagree, and the disagreement itself is informative:

  GINI / MDI IMPORTANCE (Mean Decrease in Impurity)
    Comes for free from the trained Random Forest - it's the average
    amount each feature reduced Gini impurity, averaged across every
    split, across every tree. Fast, but has a known bias: it tends to
    over-rate continuous, high-cardinality features (like our RR
    intervals, which can take almost any value) over features with
    more repetitive/discrete-ish value patterns, REGARDLESS of whether
    that feature is actually more useful for genuinely NEW predictions.

  PERMUTATION IMPORTANCE
    Takes the ALREADY-TRAINED model, shuffles one feature column at a
    time in the TEST set (breaking that feature's real relationship
    with the outcome while leaving every other feature untouched), and
    measures how much performance drops. A feature the model actually
    relies on will hurt performance badly when scrambled; a feature it
    barely uses won't move the score much. This is slower (it has to
    re-evaluate the model many times) but far less biased than Gini
    importance, and - because it's computed on the TEST set - it
    reflects real, generalizable reliance rather than a pattern the
    model may have only picked up on the training data.

Scoring metric for permutation importance: we deliberately use
'f1_macro' (which treats every class equally) rather than the default
accuracy. Given how heavily N dominates this dataset, accuracy-based
importance would mostly just describe "what helps predict N" - f1_macro
keeps the measurement aligned with what this whole project actually
cares about: performance across ALL 5 classes, not just the majority one.
"""

import numpy as np
import pandas as pd
import joblib
import smote_utils  # noqa: F401 - must be imported so joblib can find build_sampling_strategy when unpickling the model
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

FEATURE_COLUMNS = [
    "pre_rr", "post_rr", "local_avg_rr",
    "qrs_duration", "pr_interval", "qt_interval", "p_amplitude",
    "skewness", "kurtosis",
]

# Set to False when running the full pipeline back to back - with it
# True, this script pops up a plot window and PAUSES until you close
# it, which looks like a freeze if you're not expecting it (same issue
# Step 2 had). Set True if running this file on its own and you want to
# see the plot immediately.
SHOW_PLOTS = False

# The specific features H1 predicted would dominate - checked directly
# against the actual rankings below.
RR_INTERVAL_FEATURES = ["pre_rr", "post_rr", "local_avg_rr"]
H1_NAMED_FEATURES = ["pre_rr", "post_rr", "local_avg_rr", "qrs_duration", "p_amplitude"]


def compute_gini_importance(model, feature_names) -> pd.DataFrame:
    """
    NOTE for the gridsearch/pipeline version: `model` here is a full
    [SMOTE -> RandomForest] imblearn Pipeline, not a bare
    RandomForestClassifier - feature_importances_ lives on the
    classifier STEP inside it, accessed via model.named_steps, not on
    the pipeline object directly.
    """
    """
    Pulls the Random Forest's built-in Mean Decrease in Impurity (MDI)
    importance - already computed during training, no extra work needed.
    """
    importances = model.named_steps["classifier"].feature_importances_
    df = pd.DataFrame({"feature": feature_names, "gini_importance": importances})
    return df.sort_values("gini_importance", ascending=False).reset_index(drop=True)


def compute_permutation_importance(model, X_test, y_test, feature_names,
                                    n_repeats: int = 20, scoring: str = "f1_macro") -> pd.DataFrame:
    """
    Runs permutation importance on the TEST set. n_repeats=20 means
    each feature gets shuffled and re-evaluated 20 times (with
    different random shuffles), so we get a mean AND a standard
    deviation for each feature's importance - a feature with a small
    std is a consistently important one; a large std means its
    apparent importance is noisy and less trustworthy.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=42, scoring=scoring, n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "permutation_importance_mean": result.importances_mean,
        "permutation_importance_std": result.importances_std,
    })
    return df.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)


def plot_importance_comparison(gini_df: pd.DataFrame, perm_df: pd.DataFrame) -> None:
    """
    Plots both importance rankings side by side as horizontal bar
    charts, so it's immediately visible whether the two methods agree
    on which features matter most.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    gini_sorted = gini_df.sort_values("gini_importance")
    axes[0].barh(gini_sorted["feature"], gini_sorted["gini_importance"], color="#61223B")
    axes[0].set_title("Gini / MDI Importance")
    axes[0].set_xlabel("Mean decrease in impurity")

    perm_sorted = perm_df.sort_values("permutation_importance_mean")
    axes[1].barh(perm_sorted["feature"], perm_sorted["permutation_importance_mean"],
                 xerr=perm_sorted["permutation_importance_std"], color="#B79962")
    axes[1].set_title("Permutation Importance (f1_macro)")
    axes[1].set_xlabel("Mean drop in f1_macro when shuffled")

    plt.tight_layout()
    plt.savefig("feature_importance_gridsearch.png", dpi=150)
    plt.show() if SHOW_PLOTS else plt.close(fig)
    print("Saved feature_importance_relaxed.png")


def check_against_h1(gini_df: pd.DataFrame, perm_df: pd.DataFrame) -> None:
    """
    Directly checks H1's specific biomarker claim - that RR-interval
    and morphological features (QRS duration, P-wave amplitude) would
    be the dominant predictors - against what the two importance
    methods actually found.
    """
    print("\n" + "=" * 70)
    print("CHECKING AGAINST H1's BIOMARKER CLAIM")
    print("=" * 70)
    print("H1 predicted RR-interval features (pre_rr, post_rr, local_avg_rr),")
    print("QRS duration, and P-wave amplitude would be the dominant predictors.\n")

    for name, df, col in [("Gini/MDI", gini_df, "gini_importance"),
                           ("Permutation", perm_df, "permutation_importance_mean")]:
        ranks = {row["feature"]: i + 1 for i, row in df.iterrows()}
        print(f"{name} ranking (1 = most important):")
        for feature in H1_NAMED_FEATURES:
            print(f"  {feature:<15} rank {ranks[feature]} of {len(df)}")
        top3 = df.head(3)["feature"].tolist()
        print(f"  Top 3 overall: {top3}\n")


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Loading the tuned [SMOTE -> RandomForest] pipeline. permutation_importance()
    # works directly on the whole pipeline with no changes needed - it
    # shuffles the INPUT features and calls .predict(), which correctly
    # skips the SMOTE step (SMOTE only ever acts during .fit()).
    model = joblib.load("random_forest_model_gridsearch.joblib")
    X_test = pd.read_csv("X_test_relaxed.csv")
    y_test = pd.read_csv("y_test_relaxed.csv").squeeze("columns")

    print("Computing Gini/MDI importance...")
    gini_df = compute_gini_importance(model, FEATURE_COLUMNS)
    print(gini_df.round(4).to_string(index=False))

    print("\nComputing permutation importance (this takes a bit longer - "
          "it re-evaluates the model 20 times per feature)...")
    perm_df = compute_permutation_importance(model, X_test, y_test, FEATURE_COLUMNS)
    print(perm_df.round(4).to_string(index=False))

    check_against_h1(gini_df, perm_df)

    plot_importance_comparison(gini_df, perm_df)

    gini_df.to_csv("gini_importance_gridsearch.csv", index=False)
    perm_df.to_csv("permutation_importance_gridsearch.csv", index=False)
    print("\nSaved gini_importance_gridsearch.csv, permutation_importance_gridsearch.csv")
