# -*- coding: utf-8 -*-
"""
AutoGluon native feature pruning experiment with feature_prune_kwargs.

Put this file next to:
    8_autogluon_regressor_selection_per_problem_normalized.py

This script calls:
    predictor.fit(..., feature_prune_kwargs={})

It still uses external GroupKFold by problem and evaluates with algorithm-selection metrics.

Outputs:
    data/Combined/autogluon_native_feature_prune/
"""

import os

# Limit BLAS/OpenMP threads before importing numpy / sklearn backends.
# This avoids OpenBLAS NUM_THREADS / BLAS memory unallocation issues,
# especially when AutoGluon tries KNN or other sklearn-based models.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import json
import time
import shutil
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

BASE_SCRIPT = "8_autogluon_regressor_selection_per_problem_normalized_strict_no_ray.py"
OUT_DIR = "data/Combined/autogluon_native_feature_prune"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
PREDICTOR_DIR = os.path.join(OUT_DIR, "autogluon_predictors")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PREDICTOR_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
AG_PRESETS = "medium_quality"
AG_TIME_LIMIT_PER_FOLD = 1200
AG_TIME_LIMIT_FINAL = 3600
AG_EVAL_METRIC = "mean_absolute_error"
FEATURE_PRUNE_KWARGS = {}  # enables AutoGluon default feature pruning
# Alternative for some AutoGluon versions: FEATURE_PRUNE_KWARGS = {"force_prune": True}
FIT_FINAL_MODEL = True

# Strict no-Ray / no-bag / no-stack settings.
AG_FIT_STRATEGY = "sequential"
AG_DYNAMIC_STACKING = False
AG_AUTO_STACK = False
AG_NUM_CPUS = 24
AG_NUM_GPUS = 1
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


def fit_autogluon_with_pruning(train_ag, path, time_limit):
    TabularPredictor = import_autogluon()
    if os.path.exists(path):
        shutil.rmtree(path)
    predictor = TabularPredictor(label="target_auc", problem_type="regression", eval_metric=AG_EVAL_METRIC, path=path, verbosity=2)
    predictor.fit(
        train_data=train_ag,
        presets=AG_PRESETS,
        hyperparameters=AG_HYPERPARAMETERS,
        feature_prune_kwargs=FEATURE_PRUNE_KWARGS,
        time_limit=time_limit,
        num_cpus=AG_NUM_CPUS,
        num_gpus=AG_NUM_GPUS,
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


def summarize_metrics(test_meta, pred, selector_df, fold):
    y = test_meta["target_auc"].to_numpy(float)
    p = np.asarray(pred, dtype=float)
    return {
        "fold": fold,
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


def get_predictor_info(predictor):
    info = {}
    for name, func in {
        "features": lambda: predictor.features(),
        "feature_metadata": lambda: str(predictor.feature_metadata),
        "model_names": lambda: predictor.model_names(),
    }.items():
        try:
            info[name] = func()
        except Exception as e:
            info[name + "_error"] = repr(e)
    return info



def extract_model_used_features(predictor, train_ag, fold):
    """
    Extract model-specific used/pruned features after AutoGluon native feature pruning.

    AutoGluon's pruning is model-specific. Some models may have names such as
    LightGBM_Prune, ExtraTrees_Prune, etc. This function tries several access
    methods because AutoGluon versions differ in where they store the feature list.
    """
    label = "target_auc"
    original_features = [c for c in train_ag.columns if c != label]
    train_X = train_ag.drop(columns=[label]).copy()

    try:
        model_names = predictor.model_names()
    except Exception:
        try:
            model_names = predictor.get_model_names()
        except Exception:
            model_names = []

    used_rows = []
    summary_rows = []

    for model_name in model_names:
        used = None
        method = None

        try:
            model_obj = predictor._trainer.load_model(model_name)
            if hasattr(model_obj, "features") and model_obj.features is not None:
                used = list(model_obj.features)
                method = "predictor._trainer.load_model(model).features"
        except Exception:
            pass

        if used is None:
            try:
                info = predictor.info()
                mi = info.get("model_info", {}).get(model_name, {})
                for key in ["features", "feature_names", "features_in"]:
                    if key in mi and mi[key] is not None:
                        used = list(mi[key])
                        method = f"predictor.info()['model_info'][model]['{key}']"
                        break
            except Exception:
                pass

        if used is None:
            try:
                Xt = predictor.transform_features(data=train_X, model=model_name)
                used = list(Xt.columns)
                method = "predictor.transform_features(data=X, model=model)"
            except Exception:
                pass

        if used is None:
            try:
                Xt = predictor.transform_features(model=model_name)
                used = list(Xt.columns)
                method = "predictor.transform_features(model=model)"
            except Exception:
                pass

        if used is None:
            used_rows.append({
                "fold": fold,
                "model_name": model_name,
                "extraction_method": "failed",
                "feature": None,
            })
            summary_rows.append({
                "fold": fold,
                "model_name": model_name,
                "extraction_method": "failed",
                "n_original_features": len(original_features),
                "n_used_original_features": None,
                "n_pruned_original_features": None,
                "used_original_features": "",
                "pruned_original_features": "",
            })
            continue

        used = [str(f) for f in used]
        for f in used:
            used_rows.append({
                "fold": fold,
                "model_name": model_name,
                "extraction_method": method,
                "feature": f,
            })

        used_original = sorted(set(used) & set(original_features))
        pruned_original = sorted(set(original_features) - set(used_original))

        summary_rows.append({
            "fold": fold,
            "model_name": model_name,
            "extraction_method": method,
            "n_original_features": len(original_features),
            "n_used_original_features": len(used_original),
            "n_pruned_original_features": len(pruned_original),
            "used_original_features": ";".join(used_original),
            "pruned_original_features": ";".join(pruned_original),
        })

    return pd.DataFrame(used_rows), pd.DataFrame(summary_rows)


def save_pruned_feature_outputs(predictor, train_ag, fold, out_dir):
    used_df, summary_df = extract_model_used_features(predictor, train_ag, fold)

    used_path = os.path.join(out_dir, f"native_feature_prune_used_features_fold_{fold}.csv")
    summary_path = os.path.join(out_dir, f"native_feature_prune_pruned_features_summary_fold_{fold}.csv")

    used_df.to_csv(used_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved: {used_path}")
    print(f"Saved: {summary_path}")

    return used_df, summary_df


def plot_summary(leaderboard):
    for metric, ylabel, higher_is_better in [
        ("mean_normalized_regret_vs_oracle", "Mean normalized regret vs oracle", False),
        ("oracle_match_accuracy", "Oracle match accuracy", True),
        ("mean_top3_overlap", "Top-3 overlap", True),
        ("regressor_spearman", "Regressor Spearman", True),
        ("regressor_mae", "Regressor MAE", False),
    ]:
        val = float(leaderboard[metric].iloc[0])
        plt.figure(figsize=(5, 5))
        plt.bar(["AG feature prune"], [val])
        plt.ylabel(ylabel)
        plt.title(f"AutoGluon native feature prune: {metric}")
        plt.grid(axis="y", alpha=0.25)
        plt.text(0, val, f"{val:.4g}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"native_feature_prune_{metric}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved: {path}")


def plot_pred_vs_true(pred_df):
    y = pred_df["target_auc"].to_numpy(float)
    p = pred_df["pred_target_auc"].to_numpy(float)
    lo = float(np.nanmin([y.min(), p.min()]))
    hi = float(np.nanmax([y.max(), p.max()]))
    pad = 0.05 * (hi - lo + 1e-12)
    plt.figure(figsize=(7, 6))
    for source, sub in pred_df.groupby("problem_type"):
        plt.scatter(sub["target_auc"], sub["pred_target_auc"], s=18, alpha=0.50, label=f"{source} (n={len(sub)})")
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    plt.xlabel("True per-problem-normalized target")
    plt.ylabel("Predicted target")
    plt.title("AutoGluon native feature prune: predicted vs true")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "native_feature_prune_pred_vs_true_by_source.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def main():
    base = load_base_module()
    print("=== Loading base data ===")
    ela_df, perf_df = base.load_all_data()
    normalizer_params = base.fit_problem_normalizers(perf_df)
    perf_df = base.add_problem_normalized_target(perf_df, normalizer_params)
    feature_cols = base.get_feature_cols(ela_df)
    ag_df, meta_df = base.build_training_table(ela_df, perf_df, feature_cols)
    groups = base.make_cv_group(meta_df)
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    print(f"Rows={len(ag_df)}, groups={len(np.unique(groups))}, ELA features before pruning={len(feature_cols)}, algorithms={meta_df['algname'].nunique()}")

    all_metrics, all_predictions, all_selectors, all_infos, all_leaderboards = [], [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(ag_df, ag_df["target_auc"], groups)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        start = time.time()
        train_ag = ag_df.iloc[train_idx].copy()
        test_ag = ag_df.iloc[test_idx].copy()
        train_meta = meta_df.iloc[train_idx].copy()
        test_meta = meta_df.iloc[test_idx].copy()
        predictor_path = os.path.join(PREDICTOR_DIR, f"fold_{fold}")
        predictor = fit_autogluon_with_pruning(train_ag, predictor_path, AG_TIME_LIMIT_PER_FOLD)
        try:
            lb = predictor.leaderboard(test_ag, silent=True)
            lb["fold"] = fold
            all_leaderboards.append(lb)
            lb.to_csv(os.path.join(OUT_DIR, f"native_feature_prune_internal_leaderboard_fold_{fold}.csv"), index=False)
        except Exception as e:
            print(f"[Warning] leaderboard failed: {e}")
        info = get_predictor_info(predictor)
        info["fold"] = fold
        all_infos.append(info)
        pred = predictor.predict(test_ag.drop(columns=["target_auc"])).to_numpy(dtype=float)
        pred_df = test_meta.copy()
        pred_df["fold"] = fold
        pred_df["model"] = "AutoGluon_NativeFeaturePrune"
        pred_df["pred_target_auc"] = pred
        pred_df["error"] = pred_df["pred_target_auc"] - pred_df["target_auc"]
        pred_df["abs_error"] = np.abs(pred_df["error"])
        all_predictions.append(pred_df)
        selector_df = base.selector_validation(test_meta, pred, train_meta, fold)
        selector_df["model"] = "AutoGluon_NativeFeaturePrune"
        all_selectors.append(selector_df)
        row = summarize_metrics(test_meta, pred, selector_df, fold)
        row["elapsed_seconds"] = time.time() - start
        all_metrics.append(row)
        print(f"regret={row['mean_normalized_regret_vs_oracle']:.4g}, oracle_match={row['oracle_match_accuracy']:.4g}, top3={row['mean_top3_overlap']:.4g}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUT_DIR, "native_feature_prune_metrics_by_fold.csv"), index=False)
    leaderboard = pd.DataFrame([{
        "model": "AutoGluon_NativeFeaturePrune",
        "regressor_mae": metrics_df["regressor_mae"].mean(),
        "regressor_rmse": metrics_df["regressor_rmse"].mean(),
        "regressor_r2": metrics_df["regressor_r2"].mean(),
        "regressor_spearman": metrics_df["regressor_spearman"].mean(),
        "oracle_match_accuracy": metrics_df["oracle_match_accuracy"].mean(),
        "mean_normalized_regret_vs_oracle": metrics_df["mean_normalized_regret_vs_oracle"].mean(),
        "median_normalized_regret_vs_oracle": metrics_df["median_normalized_regret_vs_oracle"].mean(),
        "mean_fixed_baseline_normalized_regret_vs_oracle": metrics_df["mean_fixed_baseline_normalized_regret_vs_oracle"].mean(),
        "mean_normalized_regret_improvement_over_fixed_baseline": metrics_df["mean_normalized_regret_improvement_over_fixed_baseline"].mean(),
        "mean_ranking_spearman": metrics_df["mean_ranking_spearman"].mean(),
        "mean_top3_overlap": metrics_df["mean_top3_overlap"].mean(),
        "elapsed_seconds_mean": metrics_df["elapsed_seconds"].mean(),
    }])
    leaderboard.to_csv(os.path.join(OUT_DIR, "native_feature_prune_leaderboard.csv"), index=False)
    pred_all = pd.concat(all_predictions, ignore_index=True)
    selector_all = pd.concat(all_selectors, ignore_index=True)
    pred_all.to_csv(os.path.join(OUT_DIR, "native_feature_prune_predictions_all_folds.csv"), index=False)
    selector_all.to_csv(os.path.join(OUT_DIR, "native_feature_prune_selector_all_folds.csv"), index=False)
    with open(os.path.join(OUT_DIR, "native_feature_prune_fit_info.json"), "w") as f:
        json.dump(all_infos, f, indent=2, default=str)
    if all_leaderboards:
        pd.concat(all_leaderboards, ignore_index=True).to_csv(os.path.join(OUT_DIR, "native_feature_prune_internal_leaderboards_all_folds.csv"), index=False)
    plot_summary(leaderboard)
    plot_pred_vs_true(pred_all)

    if FIT_FINAL_MODEL:
        print("\n=== Fitting final native feature-prune model on all data ===")
        final_path = os.path.join(PREDICTOR_DIR, "final_all_data")
        final_predictor = fit_autogluon_with_pruning(ag_df, final_path, AG_TIME_LIMIT_FINAL)
        with open(os.path.join(OUT_DIR, "native_feature_prune_final_fit_info.json"), "w") as f:
            json.dump(get_predictor_info(final_predictor), f, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "run_config.json"), "w") as f:
        json.dump({"base_script": BASE_SCRIPT, "feature_prune_kwargs": FEATURE_PRUNE_KWARGS,
                   "ag_presets": AG_PRESETS, "primary_metric": "mean_normalized_regret_vs_oracle"}, f, indent=2)
    print("\nDone.")
    print(leaderboard)


if __name__ == "__main__":
    main()
