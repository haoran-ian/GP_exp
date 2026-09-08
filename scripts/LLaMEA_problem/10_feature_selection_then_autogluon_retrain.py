# -*- coding: utf-8 -*-
"""
Feature selection + AutoGluon retraining for ELA-based algorithm selection.

Put this file next to:
    8_autogluon_regressor_selection_per_problem_normalized.py

Workflow per GroupKFold fold:
    1) rank ELA features on train fold only using ExtraTreesRegressor
    2) build feature subsets: FULL, TOP-k, POSITIVE_IMPORTANCE, GROUP_TOP-k, NO_PCA, NO_DISP
    3) retrain AutoGluon on each subset
    4) evaluate on held-out problems with algorithm-selection metrics

Outputs:
    data/Combined/autogluon_feature_selection_then_retrain/
"""

import os
import json
import time
import shutil
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

BASE_SCRIPT = "8_autogluon_regressor_selection_per_problem_normalized_strict_no_ray.py"
OUT_DIR = "data/Combined/autogluon_feature_selection_then_retrain"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
PREDICTOR_DIR = os.path.join(OUT_DIR, "autogluon_predictors")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PREDICTOR_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

# AutoGluon config. Kept strict no-Ray / no-bag / no-stack.
AG_PRESETS = "medium_quality"
AG_TIME_LIMIT_PER_FOLD = 900
AG_EVAL_METRIC = "mean_absolute_error"
AG_FIT_STRATEGY = "sequential"
AG_DYNAMIC_STACKING = False
AG_AUTO_STACK = False
AG_NUM_BAG_FOLDS = 0
AG_NUM_BAG_SETS = 1
AG_NUM_STACK_LEVELS = 0
AG_USE_BAG_HOLDOUT = False
AG_HYPERPARAMETERS = {
    # "GBM": {}, "CAT": {}, "XGB": {}, "RF": {}, "XT": {}, "KNN": {},
    "NN_TORCH": {},
    # "FASTAI": {},
    "FT_TRANSFORMER": {},
    "TABPFN-3": {},
    "TABICL": {},
    "TABM": {},
}

# Feature subset candidates.
TOP_K_LIST = [10, 20, 30, 40, 50]
GROUP_TOP_K_LIST = [1, 2, 3]
MIN_POSITIVE_FEATURES = 5
INCLUDE_HANDCRAFTED_GROUP_DROPS = True
RUN_SUBSETS = None  # e.g. ["FULL", "TOP_20", "GROUP_TOP_2"] for debugging

# ExtraTrees ranker.
FS_N_ESTIMATORS = 800
FS_MAX_FEATURES = "sqrt"
FS_MIN_SAMPLES_LEAF = 1
FS_N_JOBS = -1


def load_base_module():
    here = Path(__file__).resolve().parent
    path = here / BASE_SCRIPT
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {BASE_SCRIPT} in {here}. Put this script next to it.")
    spec = importlib.util.spec_from_file_location("ag_base", str(path))
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def import_autogluon():
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as e:
        raise ImportError("Install AutoGluon with: pip install autogluon.tabular") from e
    return TabularPredictor


def get_feature_group(feature_name):
    f = str(feature_name)
    if "ela_meta" in f:
        return "Meta-model"
    if "ela_distr" in f or "ela_distribution" in f:
        return "Distribution"
    if "ela_level" in f:
        return "Level-set"
    if "nbc" in f:
        return "Nearest Better"
    if "ic" in f or "information_content" in f:
        return "Info. Content"
    if "disp" in f or "dispersion" in f:
        return "Dispersion"
    if "pca" in f:
        return "PCA"
    if "limo" in f:
        return "Linear Model"
    return "Others"


def rank_features_extratrees(train_meta, feature_cols):
    X_ela = train_meta[feature_cols].copy()
    alg_dummies = pd.get_dummies(train_meta["algname"].astype(str), prefix="algname")
    X = pd.concat([X_ela.reset_index(drop=True), alg_dummies.reset_index(drop=True)], axis=1)
    y = train_meta["target_auc"].to_numpy(dtype=float)
    model = ExtraTreesRegressor(
        n_estimators=FS_N_ESTIMATORS,
        max_features=FS_MAX_FEATURES,
        min_samples_leaf=FS_MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=FS_N_JOBS,
    )
    model.fit(X, y)
    imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    imp = imp[imp["feature"].isin(feature_cols)].copy()
    imp["group"] = imp["feature"].apply(get_feature_group)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def construct_feature_subsets(importance_df, all_feature_cols):
    subsets = {"FULL": list(all_feature_cols)}
    for k in TOP_K_LIST:
        subsets[f"TOP_{k}"] = importance_df.head(k)["feature"].tolist()
    positive = importance_df[importance_df["importance"] > 0]["feature"].tolist()
    if len(positive) < MIN_POSITIVE_FEATURES:
        positive = importance_df.head(MIN_POSITIVE_FEATURES)["feature"].tolist()
    subsets["POSITIVE_IMPORTANCE"] = positive
    group_scores = (importance_df.groupby("group", as_index=False)
                    .agg(group_importance=("importance", "sum"), n_features=("feature", "size"))
                    .sort_values("group_importance", ascending=False))
    for k in GROUP_TOP_K_LIST:
        groups = group_scores.head(k)["group"].tolist()
        subsets[f"GROUP_TOP_{k}"] = importance_df[importance_df["group"].isin(groups)]["feature"].tolist()
    if INCLUDE_HANDCRAFTED_GROUP_DROPS:
        subsets["NO_PCA"] = [f for f in all_feature_cols if get_feature_group(f) != "PCA"]
        subsets["NO_DISP"] = [f for f in all_feature_cols if get_feature_group(f) != "Dispersion"]
        subsets["NO_PCA_NO_DISP"] = [f for f in all_feature_cols if get_feature_group(f) not in {"PCA", "Dispersion"}]
    subsets = {k: sorted(set(v)) for k, v in subsets.items() if len(v) > 0}
    if RUN_SUBSETS is not None:
        subsets = {k: v for k, v in subsets.items() if k in set(RUN_SUBSETS)}
    return subsets, group_scores


def make_autogluon_table_from_meta(meta_df, selected_features):
    cols = list(selected_features) + ["algname", "target_auc"]
    if "dim" in meta_df.columns:
        cols.insert(0, "dim")
    df = meta_df[cols].copy()
    df["algname"] = df["algname"].astype("category")
    return df


def fit_autogluon(train_ag, path, time_limit):
    TabularPredictor = import_autogluon()
    if os.path.exists(path):
        shutil.rmtree(path)
    predictor = TabularPredictor(
        label="target_auc",
        problem_type="regression",
        eval_metric=AG_EVAL_METRIC,
        path=path,
        verbosity=2,
    )
    predictor.fit(
        train_data=train_ag,
        presets=AG_PRESETS,
        hyperparameters=AG_HYPERPARAMETERS,
        time_limit=time_limit,
        num_cpus="auto",
        num_gpus="auto",
        fit_strategy=AG_FIT_STRATEGY,
        dynamic_stacking=AG_DYNAMIC_STACKING,
        auto_stack=AG_AUTO_STACK,
        num_bag_folds=AG_NUM_BAG_FOLDS,
        num_bag_sets=AG_NUM_BAG_SETS,
        num_stack_levels=AG_NUM_STACK_LEVELS,
        use_bag_holdout=AG_USE_BAG_HOLDOUT,
    )
    return predictor


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    return np.nan if len(np.unique(y_true)) <= 1 else float(r2_score(y_true, y_pred))


def safe_spearman(y_true, y_pred):
    return np.nan if len(y_true) < 3 else float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def summarize_metrics(test_meta, pred, selector_df, fold, subset_name, n_features):
    y = test_meta["target_auc"].to_numpy(float)
    p = np.asarray(pred, dtype=float)
    return {
        "fold": fold,
        "feature_subset": subset_name,
        "n_features": n_features,
        "regressor_mae": float(mean_absolute_error(y, p)),
        "regressor_rmse": rmse(y, p),
        "regressor_r2": safe_r2(y, p),
        "regressor_spearman": safe_spearman(y, p),
        "oracle_match_accuracy": float(selector_df["selected_is_oracle"].mean()),
        "mean_normalized_regret_vs_oracle": float(selector_df["normalized_regret_vs_oracle"].mean()),
        "median_normalized_regret_vs_oracle": float(selector_df["normalized_regret_vs_oracle"].median()),
        "mean_fixed_baseline_normalized_regret_vs_oracle": float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean()),
        "mean_normalized_regret_improvement_over_fixed_baseline": float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean() - selector_df["normalized_regret_vs_oracle"].mean()),
        "mean_ranking_spearman": float(selector_df["ranking_spearman"].mean()),
        "mean_top3_overlap": float(selector_df["top3_overlap"].mean()),
    }


def plot_leaderboard(leaderboard):
    for metric, ylabel, higher_is_better in [
        ("mean_normalized_regret_vs_oracle", "Mean normalized regret vs oracle", False),
        ("oracle_match_accuracy", "Oracle match accuracy", True),
        ("mean_top3_overlap", "Top-3 overlap", True),
        ("regressor_spearman", "Regressor Spearman", True),
        ("regressor_mae", "Regressor MAE", False),
    ]:
        df = leaderboard.dropna(subset=[metric]).sort_values(metric, ascending=not higher_is_better)
        plt.figure(figsize=(10, max(5, 0.35 * len(df))))
        plt.barh(df["feature_subset"], df[metric])
        plt.xlabel(ylabel)
        plt.ylabel("Feature subset")
        plt.title(f"AutoGluon with selected ELA features: {metric}")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"feature_subset_{metric}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved: {path}")


def main():
    base = load_base_module()
    print("=== Loading base data ===")
    ela_df, perf_df = base.load_all_data()
    normalizer_params = base.fit_problem_normalizers(perf_df)
    perf_df = base.add_problem_normalized_target(perf_df, normalizer_params)
    all_feature_cols = base.get_feature_cols(ela_df)
    _, meta_df = base.build_training_table(ela_df, perf_df, all_feature_cols)
    groups = base.make_cv_group(meta_df)
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    print(f"Rows={len(meta_df)}, groups={len(np.unique(groups))}, ELA features={len(all_feature_cols)}, algorithms={meta_df['algname'].nunique()}")

    all_metrics, all_predictions, all_selectors = [], [], []
    all_selected_features, all_importances, all_group_scores = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(meta_df, meta_df["target_auc"], groups)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        train_meta = meta_df.iloc[train_idx].copy()
        test_meta = meta_df.iloc[test_idx].copy()
        imp = rank_features_extratrees(train_meta, all_feature_cols)
        imp["fold"] = fold
        all_importances.append(imp)
        imp.to_csv(os.path.join(OUT_DIR, f"feature_importance_fold_{fold}.csv"), index=False)
        subsets, group_scores = construct_feature_subsets(imp, all_feature_cols)
        group_scores["fold"] = fold
        all_group_scores.append(group_scores)
        group_scores.to_csv(os.path.join(OUT_DIR, f"feature_group_importance_fold_{fold}.csv"), index=False)

        for subset_name, selected_features in subsets.items():
            print(f"\n--- fold={fold}, subset={subset_name}, n_features={len(selected_features)} ---")
            start = time.time()
            all_selected_features.extend([
                {"fold": fold, "feature_subset": subset_name, "feature": f, "group": get_feature_group(f)}
                for f in selected_features
            ])
            train_ag = make_autogluon_table_from_meta(train_meta, selected_features)
            test_ag = make_autogluon_table_from_meta(test_meta, selected_features)
            predictor_path = os.path.join(PREDICTOR_DIR, f"fold_{fold}_{subset_name}")
            predictor = fit_autogluon(train_ag, predictor_path, AG_TIME_LIMIT_PER_FOLD)
            pred = predictor.predict(test_ag.drop(columns=["target_auc"])).to_numpy(dtype=float)
            pred_df = test_meta.copy()
            pred_df["fold"] = fold
            pred_df["feature_subset"] = subset_name
            pred_df["pred_target_auc"] = pred
            pred_df["error"] = pred_df["pred_target_auc"] - pred_df["target_auc"]
            pred_df["abs_error"] = np.abs(pred_df["error"])
            all_predictions.append(pred_df)
            selector_df = base.selector_validation(test_meta, pred, train_meta, fold)
            selector_df["feature_subset"] = subset_name
            all_selectors.append(selector_df)
            row = summarize_metrics(test_meta, pred, selector_df, fold, subset_name, len(selected_features))
            row["elapsed_seconds"] = time.time() - start
            all_metrics.append(row)
            print(f"regret={row['mean_normalized_regret_vs_oracle']:.4g}, oracle_match={row['oracle_match_accuracy']:.4g}, top3={row['mean_top3_overlap']:.4g}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUT_DIR, "feature_subset_metrics_by_fold.csv"), index=False)
    leaderboard = (metrics_df.groupby("feature_subset", as_index=False)
        .agg(n_features_mean=("n_features", "mean"), regressor_mae=("regressor_mae", "mean"),
             regressor_rmse=("regressor_rmse", "mean"), regressor_r2=("regressor_r2", "mean"),
             regressor_spearman=("regressor_spearman", "mean"), oracle_match_accuracy=("oracle_match_accuracy", "mean"),
             mean_normalized_regret_vs_oracle=("mean_normalized_regret_vs_oracle", "mean"),
             median_normalized_regret_vs_oracle=("median_normalized_regret_vs_oracle", "mean"),
             mean_fixed_baseline_normalized_regret_vs_oracle=("mean_fixed_baseline_normalized_regret_vs_oracle", "mean"),
             mean_normalized_regret_improvement_over_fixed_baseline=("mean_normalized_regret_improvement_over_fixed_baseline", "mean"),
             mean_ranking_spearman=("mean_ranking_spearman", "mean"), mean_top3_overlap=("mean_top3_overlap", "mean"),
             elapsed_seconds_mean=("elapsed_seconds", "mean"))
        .sort_values(["mean_normalized_regret_vs_oracle", "oracle_match_accuracy", "mean_top3_overlap"], ascending=[True, False, False]))
    leaderboard.to_csv(os.path.join(OUT_DIR, "feature_subset_leaderboard.csv"), index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(os.path.join(OUT_DIR, "feature_subset_predictions_all_folds.csv"), index=False)
    pd.concat(all_selectors, ignore_index=True).to_csv(os.path.join(OUT_DIR, "feature_subset_selector_all_folds.csv"), index=False)
    pd.DataFrame(all_selected_features).to_csv(os.path.join(OUT_DIR, "selected_features_by_fold_and_subset.csv"), index=False)
    pd.concat(all_importances, ignore_index=True).to_csv(os.path.join(OUT_DIR, "feature_importance_all_folds.csv"), index=False)
    pd.concat(all_group_scores, ignore_index=True).to_csv(os.path.join(OUT_DIR, "feature_group_importance_all_folds.csv"), index=False)
    plot_leaderboard(leaderboard)
    with open(os.path.join(OUT_DIR, "run_config.json"), "w") as f:
        json.dump({"base_script": BASE_SCRIPT, "top_k_list": TOP_K_LIST, "group_top_k_list": GROUP_TOP_K_LIST,
                   "ag_presets": AG_PRESETS, "primary_metric": "mean_normalized_regret_vs_oracle"}, f, indent=2)
    print("\nDone. Best subsets:")
    print(leaderboard.head(10))


if __name__ == "__main__":
    main()
