"""
STEP 6 — Random Forest + SMOTE, tuned via GridSearchCV (leakage-safe)
==========================================================================
Combines SMOTE and the Random Forest into ONE imblearn Pipeline, then
runs a grid search with 5-fold stratified cross-validation over that
WHOLE pipeline - not over the Random Forest alone.

WHY THIS HAS TO BE ONE COMBINED STEP, NOT TWO SEPARATE ONES: if SMOTE
were applied once to the full training set and cross-validation run
afterward, a synthetic point generated from a real beat that lands in
a given fold's VALIDATION portion could end up training that same
fold - leaking information across the train/validation boundary. Using
imblearn's Pipeline (NOT scikit-learn's own Pipeline - sklearn's
version can't handle a step that changes the number of samples)
guarantees SMOTE is refit from scratch on ONLY each fold's own training
portion, every single time, and is automatically skipped entirely at
prediction time (so it never touches the test set, same as before).

Install what this script needs:
    pip install scikit-learn imbalanced-learn joblib
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# IMPORTANT: build_sampling_strategy must live in its own shared file
# (smote_utils.py), NOT here, so that a saved model can still find it
# when loaded from a DIFFERENT script (e.g. step7/step8). See
# smote_utils.py's docstring for the full explanation of why.
from smote_utils import build_sampling_strategy

# A FIXED, conservative k_neighbors, rather than the fold-adaptive value
# used in the single-shot SMOTE script. Reasoning: GridSearchCV fits
# SMOTE fresh inside each of the 5 folds, on only ~80% of the training
# data each time - our rarest class (Q, 8 total real examples) could
# shrink to as few as 6 in a given fold's training portion. k_neighbors
# must be chosen BEFORE fitting (it can't adapt per-fold the way our
# single-shot choose_k_neighbors() function could), so we use a smaller,
# safely-conservative k=3 here rather than risk a fold where k=5 isn't
# achievable and the whole grid search crashes partway through.
SMOTE_K_NEIGHBORS = 3


def build_pipeline() -> Pipeline:
    """
    The combined [SMOTE -> RandomForest] pipeline that GridSearchCV
    will tune as a single unit. Hyperparameter names are prefixed with
    "classifier__" - imblearn/scikit-learn's convention for addressing
    a specific step's parameters inside a Pipeline/GridSearchCV.
    """
    smote = SMOTE(
        sampling_strategy=build_sampling_strategy,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=42,
    )
    classifier = RandomForestClassifier(
        max_features="sqrt",   # kept fixed - not part of the grid, matches prior methodology
        bootstrap=True,
        class_weight=None,     # SMOTE already handles balancing - avoid double-correcting
        random_state=42,
        n_jobs=1,               # deliberately NOT -1 here - see note in __main__ about nested parallelism
    )
    return Pipeline([
        ("smote", smote),
        ("classifier", classifier),
    ])


def log_fold_composition(X_train, y_train, cv) -> pd.DataFrame:
    """
    Logs the ACTUAL per-fold class counts and SMOTE targets, by
    manually replaying the same StratifiedKFold split GridSearchCV will
    use internally - purely for reporting purposes, this does not
    affect the grid search itself in any way.

    This replaces guessing/approximating per-fold numbers (e.g. "80% of
    930 is about 744") with the real counts each fold actually
    contains, and the real SMOTE target each fold's own
    build_sampling_strategy() call would produce.
    """
    from collections import Counter

    rows = []
    for fold_num, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        fold_y_train = y_train.iloc[train_idx]
        fold_y_val = y_train.iloc[val_idx]

        real_counts = Counter(fold_y_train)
        smote_targets = build_sampling_strategy(fold_y_train)

        for cls in sorted(set(real_counts) | set(smote_targets)):
            rows.append({
                "fold": fold_num,
                "class": cls,
                "real_count_in_fold_training_portion": real_counts.get(cls, 0),
                "smote_target_for_this_fold": smote_targets.get(cls, real_counts.get(cls, 0)),
                "count_in_fold_validation_portion": sum(fold_y_val == cls),
            })

    log_df = pd.DataFrame(rows)
    print("\nActual per-fold class composition (logged, not approximated):")
    print(log_df.to_string(index=False))

    log_df.to_csv("fold_composition_log.csv", index=False)
    print("\nSaved fold_composition_log.csv\n")

    return log_df


if __name__ == "__main__":
    X_train = pd.read_csv("X_train_relaxed.csv")   # NOTE: the PRE-SMOTE file - SMOTE now lives inside the pipeline
    y_train = pd.read_csv("y_train_relaxed.csv").squeeze("columns")

    pipeline = build_pipeline()

    param_grid = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [10, 20, 30, None],
        "classifier__min_samples_split": [2, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Log the REAL per-fold numbers before the actual (slow) grid search
    # starts - this is quick (just counting, not training anything) and
    # gives exact, reportable figures rather than approximations.
    log_fold_composition(X_train, y_train, cv)

    # n_jobs here controls how many (pipeline, fold) combinations run at
    # once. Each Random Forest inside is already set to n_jobs=1 above,
    # specifically to avoid NESTED parallelism (GridSearchCV spawning
    # multiple processes, each of which also tries to spawn its own
    # per-tree processes) - this exact kind of nested multiprocessing is
    # what caused the earlier MemoryError/OpenBLAS crashes in this
    # project. n_jobs=2 here is a conservative starting point; raise it
    # only if you have confirmed spare RAM (watch Task Manager), and
    # drop it to 1 immediately if you see memory errors again.
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="recall_macro",  # matches this project's stated priority on recall over raw accuracy
        n_jobs=2,
        verbose=2,                # prints progress as each fold/combination completes
    )

    print(f"Starting grid search: {len(param_grid['classifier__n_estimators'])} x "
          f"{len(param_grid['classifier__max_depth'])} x "
          f"{len(param_grid['classifier__min_samples_split'])} = "
          f"{4*4*3} parameter combinations x 5 folds = {4*4*3*5} total model fits.")
    print("This will take a while - progress will print as each fit completes.\n")

    grid_search.fit(X_train, y_train)

    print("\nGrid search complete.")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validated macro recall: {grid_search.best_score_:.4f}")

    # grid_search.best_estimator_ is the full [SMOTE -> RandomForest]
    # pipeline, already refit on the ENTIRE training set using the best
    # parameters found (GridSearchCV's default refit=True behaviour).
    # This final refit is NOT a leakage risk - there is no more
    # train/validation split happening at this point, it's simply
    # building the final model to be saved and evaluated.
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, "random_forest_model_gridsearch.joblib")
    print("Saved random_forest_model_gridsearch.joblib")

    # Save the full grid search results table too - useful for your
    # report (e.g. showing how performance varied across the grid)
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv("gridsearch_cv_results.csv", index=False)
    print("Saved gridsearch_cv_results.csv (full results for all 48 parameter combinations)")
