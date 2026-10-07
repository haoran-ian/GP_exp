
# -*- coding: utf-8 -*-
"""
Use AutoGluon to select a suitable regressor for ELA-based algorithm selection.

Training data:
    BBOB + MABBOB + LLM

Target handling:
    BBOB / MABBOB: raw AUC
    LLM: abnormal AUC filtering + log1p(AUC)
    Then per-problem normalization.

Model:
    AutoGluon TabularPredictor regression

Validation:
    GroupKFold by problem group, so algorithms from the same problem are not split
    across train/test.

Main evaluation:
    1. AutoGluon validation leaderboard per fold
    2. Regression metrics:
        MAE, RMSE, R2, Spearman
    3. Algorithm-selection metrics:
        oracle_match_accuracy
        mean_normalized_regret_vs_oracle
        mean_normalized_regret_improvement_over_fixed_baseline
        top3_overlap
        per-problem ranking Spearman

Outputs:
    data/Combined/autogluon_regressor_selection_per_problem_normalized/
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
BBOB_PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"

MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
MABBOB_PERF_PATH = "data/MABBOB/mabbob_algorithm_auc_performance.csv"

LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"
LLM_PERF_PATH = "data/LLM/llm_algorithm_auc_performance.csv"

OUT_DIR = "data/Combined/autogluon_regressor_selection_per_problem_normalized"
PREDICTOR_DIR = os.path.join(OUT_DIR, "autogluon_predictors")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PREDICTOR_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


RANDOM_SEED = 42
N_SPLITS = 5

# AutoGluon settings.
# Quick test:
#   AG_PRESETS = "medium_quality"
#   AG_TIME_LIMIT_PER_FOLD = 600
#
# Stronger run:
#   AG_PRESETS = "best_quality"
#   AG_TIME_LIMIT_PER_FOLD = 1800 or 3600
AG_PRESETS = "medium_quality"
AG_TIME_LIMIT_PER_FOLD = 900
AG_TIME_LIMIT_FINAL = 3600
AG_VERBOSITY = 2
AG_EVAL_METRIC = "mean_absolute_error"

# Ray-related settings.
# Keep AutoGluon sequential and avoid Ray-based sub-fits as much as possible.
# This is useful on clusters or environments where Ray causes instability.
AG_FIT_STRATEGY = "sequential"

# Strictly avoid Ray-related paths:
# - Disable dynamic stacking sub-fits.
# - Disable bagging and stack ensembling.
# - Disable auto_stack, because it can re-enable bagging/stacking.
AG_DYNAMIC_STACKING = False
AG_AUTO_STACK = False
AG_NUM_BAG_FOLDS = 0
AG_NUM_BAG_SETS = 1
AG_NUM_STACK_LEVELS = 0
AG_USE_BAG_HOLDOUT = False
AG_DS_ARGS = {
    "memory_safe_fits": False,
    "enable_ray_logging": False,
}

AG_HYPERPARAMETERS = {
    # Keep models relatively standard and avoid experimental Ray-heavy paths.
    # You can add "NN_TORCH" back later if needed.
    "GBM": {},
    "CAT": {},
    "XGB": {},
    "RF": {},
    "XT": {},
    "KNN": {},
}

FIT_FINAL_MODEL = True

# For real-world deployment, problem_type is not known as BBOB/MABBOB/LLM,
# so default is False. dim is usually safe to include.
INCLUDE_PROBLEM_TYPE_AS_FEATURE = False
INCLUDE_DIM_AS_FEATURE = True

AUC_MIN_POSITIVE = 1e-300
LLM_AUC_ABS_MAX = 1e100
LLM_AUC_UPPER_QUANTILE = 0.995
DROP_LLM_ABNORMAL_AUC = True

NORMALIZATION_METHOD = "minmax"
NORMALIZATION_EPS = 1e-12

META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim",
    "seed", "n_samples", "instance_id", "mabbob_instance_id",
    "llm_problem_id", "selection_method", "source_dataset",
    "lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
]


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def harmonize_feature_names(df):
    rename = {}
    for c in df.columns:
        nc = c
        nc = nc.replace("ela_distribution.", "ela_distr.")
        nc = nc.replace("dispersion.", "disp.")
        nc = nc.replace("information_content.", "ic.")
        rename[c] = nc
    return df.rename(columns=rename)


def problem_key_cols():
    return ["problem_type", "fid", "iid", "dim"]


def make_problem_key(df):
    return (
        df["problem_type"].astype(str)
        + "|fid=" + df["fid"].astype(int).astype(str)
        + "|iid=" + df["iid"].astype(int).astype(str)
        + "|dim=" + df["dim"].astype(int).astype(str)
    )


def make_cv_group(df):
    groups = []
    for _, r in df.iterrows():
        if r["problem_type"] == "BBOB":
            groups.append(f"BBOB_F{int(r['fid'])}")
        else:
            groups.append(str(r["problem_key"]))
    return np.asarray(groups)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return np.nan
    return float(r2_score(y_true, y_pred))


def safe_spearman(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def drop_invalid_problem_rows(df, problem_type, stage):
    df = df.copy()
    n0 = len(df)

    if "FAILED" in df.columns:
        failed = pd.to_numeric(df["FAILED"], errors="coerce").fillna(0) != 0
        df = df.loc[~failed].copy()

    if "dim" not in df.columns:
        raise ValueError(f"{problem_type} {stage} table has no dim column.")

    dim = pd.to_numeric(df["dim"], errors="coerce")
    valid = np.isfinite(dim) & (dim > 0)
    df = df.loc[valid].copy()
    df["dim"] = dim.loc[df.index].astype(int)

    if len(df) < n0:
        print(f"[Clean] Dropped {n0 - len(df)} invalid {problem_type} {stage} rows; kept {len(df)} / {n0}.")
    return df


def ensure_ela_keys(df, problem_type):
    df = harmonize_feature_names(df.copy())
    df = drop_invalid_problem_rows(df, problem_type, stage="ELA")
    df["problem_type"] = problem_type

    if problem_type == "BBOB":
        df = df[df["fid"].between(1, 24)].copy()
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
        if "iid" not in df.columns:
            df["iid"] = 1
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").fillna(1).astype(int)
        if "problem_name" not in df.columns:
            df["problem_name"] = df["fid"].apply(lambda x: f"BBOB_F{int(x)}")
        df["problem_name"] = df["problem_name"].fillna(df["fid"].apply(lambda x: f"BBOB_F{int(x)}"))

    elif problem_type == "MABBOB":
        if "mabbob_instance_id" not in df.columns:
            if "instance_id" in df.columns:
                df["mabbob_instance_id"] = df["instance_id"]
            elif "iid" in df.columns:
                df["mabbob_instance_id"] = df["iid"]
            else:
                raise ValueError("MABBOB ELA must contain mabbob_instance_id, instance_id, or iid.")

        iid = pd.to_numeric(df["mabbob_instance_id"], errors="coerce")
        df = df.loc[np.isfinite(iid)].copy()
        df["mabbob_instance_id"] = iid.loc[df.index].astype(int)

        df["fid"] = -100
        df["iid"] = df["mabbob_instance_id"].astype(int)
        df["problem_name"] = df["iid"].apply(lambda x: f"MABBOB_{int(x)}")

    elif problem_type == "LLM":
        if "llm_problem_id" not in df.columns:
            if "iid" in df.columns:
                df["llm_problem_id"] = df["iid"]
            elif "instance_id" in df.columns:
                df["llm_problem_id"] = df["instance_id"]
            else:
                raise ValueError("LLM ELA must contain llm_problem_id, iid, or instance_id.")

        iid = pd.to_numeric(df["llm_problem_id"], errors="coerce")
        df = df.loc[np.isfinite(iid)].copy()
        df["llm_problem_id"] = iid.loc[df.index].astype(int)

        df["fid"] = -200
        df["iid"] = df["llm_problem_id"].astype(int)
        if "problem_name" not in df.columns:
            df["problem_name"] = df["iid"].apply(lambda x: f"LLM_{int(x)}")
        df["problem_name"] = df["problem_name"].fillna(df["iid"].apply(lambda x: f"LLM_{int(x)}"))

    else:
        raise ValueError(problem_type)

    df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype(int)
    df["source_dataset"] = problem_type
    return df


def clean_and_transform_auc_by_source(df, problem_type):
    df = df.copy()
    n0 = len(df)
    df["auc_mean"] = pd.to_numeric(df["auc_mean"], errors="coerce")
    df["auc_filter_reason"] = "kept"

    if problem_type in ["BBOB", "MABBOB"]:
        invalid = ~np.isfinite(df["auc_mean"])
        df.loc[invalid, "auc_filter_reason"] = "non_finite_auc"
        df.groupby(["problem_type", "auc_filter_reason"], dropna=False).size().reset_index(name="n_rows").to_csv(
            os.path.join(OUT_DIR, f"auc_filter_report_{problem_type}.csv"), index=False
        )
        df = df.loc[~invalid].copy()
        df["auc_transform"] = "raw"
        df["transformed_auc"] = df["auc_mean"].astype(float)
        print(f"[AUC clean] {problem_type}: kept finite raw AUC {len(df)} / {n0}.")
        return df

    if problem_type == "LLM":
        invalid = (~np.isfinite(df["auc_mean"])) | (df["auc_mean"] <= AUC_MIN_POSITIVE)
        df.loc[invalid, "auc_filter_reason"] = "non_finite_or_non_positive"

        if LLM_AUC_ABS_MAX is not None:
            too_large = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > LLM_AUC_ABS_MAX)
            df.loc[too_large, "auc_filter_reason"] = f"above_global_cap_{LLM_AUC_ABS_MAX:.1e}"

        if LLM_AUC_UPPER_QUANTILE is not None:
            valid_for_q = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > AUC_MIN_POSITIVE)
            if valid_for_q.any():
                q = df.loc[valid_for_q, "auc_mean"].quantile(LLM_AUC_UPPER_QUANTILE)
                if np.isfinite(q):
                    too_large_q = valid_for_q & (df["auc_mean"] > q)
                    df.loc[too_large_q, "auc_filter_reason"] = f"above_q{LLM_AUC_UPPER_QUANTILE}"

        abnormal = df["auc_filter_reason"] != "kept"
        df.groupby(["problem_type", "auc_filter_reason"], dropna=False).size().reset_index(name="n_rows").to_csv(
            os.path.join(OUT_DIR, f"auc_filter_report_{problem_type}.csv"), index=False
        )
        df.loc[abnormal].to_csv(os.path.join(OUT_DIR, f"auc_abnormal_rows_{problem_type}.csv"), index=False)

        if DROP_LLM_ABNORMAL_AUC:
            df = df.loc[~abnormal].copy()
            print(f"[AUC clean] LLM: dropped {n0 - len(df)} abnormal rows; kept {len(df)} / {n0}.")
        else:
            df = df.loc[~invalid].copy()

        df["auc_transform"] = "log1p"
        df["transformed_auc"] = np.log1p(np.maximum(df["auc_mean"].to_numpy(float), AUC_MIN_POSITIVE))
        return df

    raise ValueError(problem_type)


def ensure_perf_keys(df, problem_type):
    df = drop_invalid_problem_rows(df.copy(), problem_type, stage="performance")
    df["problem_type"] = problem_type

    if problem_type == "BBOB":
        df = df[df["fid"].between(1, 24)].copy()
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
        if "iid" not in df.columns:
            df["iid"] = 1
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").fillna(1).astype(int)
        if "problem_name" not in df.columns:
            df["problem_name"] = df["fid"].apply(lambda x: f"BBOB_F{int(x)}")

    elif problem_type == "MABBOB":
        if "mabbob_instance_id" not in df.columns:
            if "iid" in df.columns:
                df["mabbob_instance_id"] = df["iid"]
            elif "instance_id" in df.columns:
                df["mabbob_instance_id"] = df["instance_id"]
            else:
                raise ValueError("MABBOB performance must contain mabbob_instance_id, iid, or instance_id.")
        iid = pd.to_numeric(df["mabbob_instance_id"], errors="coerce")
        df = df.loc[np.isfinite(iid)].copy()
        df["mabbob_instance_id"] = iid.loc[df.index].astype(int)
        df["fid"] = -100
        df["iid"] = df["mabbob_instance_id"].astype(int)
        df["problem_name"] = df["iid"].apply(lambda x: f"MABBOB_{int(x)}")

    elif problem_type == "LLM":
        if "llm_problem_id" not in df.columns:
            if "iid" in df.columns:
                df["llm_problem_id"] = df["iid"]
            else:
                raise ValueError("LLM performance must contain llm_problem_id or iid.")
        iid = pd.to_numeric(df["llm_problem_id"], errors="coerce")
        df = df.loc[np.isfinite(iid)].copy()
        df["llm_problem_id"] = iid.loc[df.index].astype(int)
        df["fid"] = -200
        df["iid"] = df["llm_problem_id"].astype(int)
        if "problem_name" not in df.columns:
            df["problem_name"] = df["iid"].apply(lambda x: f"LLM_{int(x)}")
        df["problem_name"] = df["problem_name"].fillna(df["iid"].apply(lambda x: f"LLM_{int(x)}"))
    else:
        raise ValueError(problem_type)

    df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype(int)
    if "auc_mean" not in df.columns:
        raise ValueError(f"{problem_type} performance table has no auc_mean column.")
    df = clean_and_transform_auc_by_source(df, problem_type)
    df["source_dataset"] = problem_type
    return df


def load_all_data():
    for p in [BBOB_ELA_PATH, BBOB_PERF_PATH, MABBOB_ELA_PATH, MABBOB_PERF_PATH, LLM_ELA_PATH, LLM_PERF_PATH]:
        require_file(p)

    bbob_ela = ensure_ela_keys(pd.read_csv(BBOB_ELA_PATH), "BBOB")
    bbob_perf = ensure_perf_keys(pd.read_csv(BBOB_PERF_PATH), "BBOB")
    mabbob_ela = ensure_ela_keys(pd.read_csv(MABBOB_ELA_PATH), "MABBOB")
    mabbob_perf = ensure_perf_keys(pd.read_csv(MABBOB_PERF_PATH), "MABBOB")
    llm_ela = ensure_ela_keys(pd.read_csv(LLM_ELA_PATH), "LLM")
    llm_perf = ensure_perf_keys(pd.read_csv(LLM_PERF_PATH), "LLM")

    ela_df = pd.concat([bbob_ela, mabbob_ela, llm_ela], ignore_index=True, sort=False)
    perf_df = pd.concat([bbob_perf, mabbob_perf, llm_perf], ignore_index=True, sort=False)
    return ela_df, perf_df


def get_feature_cols(ela_df):
    excluded = set(META_COLS)
    feature_cols = []
    for c in ela_df.columns:
        if c in excluded:
            continue
        if c in ["FAILED", "ERROR"]:
            continue
        if c.endswith(".FAILED") or c.endswith(".ERROR"):
            continue
        if pd.api.types.is_numeric_dtype(ela_df[c]):
            feature_cols.append(c)

    X = ela_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in feature_cols if not X[c].isna().all()]
    if feature_cols:
        nunique = X[feature_cols].nunique(dropna=True)
        feature_cols = [c for c in feature_cols if nunique[c] > 1]
    return sorted(feature_cols)


def clean_X(X, clip_quantile=0.999, clip_abs=1e20):
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    for c in X.columns:
        s = X[c]
        finite = s[np.isfinite(s)]
        if len(finite) == 0:
            continue
        lo = finite.quantile(1.0 - clip_quantile)
        hi = finite.quantile(clip_quantile)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            X[c] = s.clip(lower=lo, upper=hi)

    X = X.clip(lower=-clip_abs, upper=clip_abs)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X = X.replace([np.inf, -np.inf], 0.0)

    float32_max = np.finfo(np.float32).max / 100.0
    X = X.clip(lower=-float32_max, upper=float32_max)
    return X.astype(np.float32)


def fit_problem_normalizers(perf_df):
    df = perf_df.copy()
    df["problem_key"] = make_problem_key(df)
    params = {}

    for key, sub in df.groupby("problem_key"):
        values = sub["transformed_auc"].to_numpy(dtype=float)
        if NORMALIZATION_METHOD == "minmax":
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            scale = vmax - vmin
            if not np.isfinite(scale) or scale <= NORMALIZATION_EPS:
                scale = 1.0
            params[key] = {"method": "minmax", "min": vmin, "max": vmax, "scale": scale}
        elif NORMALIZATION_METHOD == "zscore":
            mean = float(np.mean(values))
            std = float(np.std(values))
            if not np.isfinite(std) or std <= NORMALIZATION_EPS:
                std = 1.0
            params[key] = {"method": "zscore", "mean": mean, "std": std}
        else:
            raise ValueError(NORMALIZATION_METHOD)
    return params


def transform_value_by_problem(value, problem_key, normalizer_params):
    p = normalizer_params[problem_key]
    value = np.asarray(value, dtype=float)
    if p["method"] == "minmax":
        return (value - p["min"]) / p["scale"]
    if p["method"] == "zscore":
        return (value - p["mean"]) / p["std"]
    raise ValueError(p["method"])


def add_problem_normalized_target(perf_df, normalizer_params):
    df = perf_df.copy()
    df["problem_key"] = make_problem_key(df)
    df["target_auc"] = np.nan
    for key, idx in df.groupby("problem_key").groups.items():
        df.loc[idx, "target_auc"] = transform_value_by_problem(
            df.loc[idx, "transformed_auc"].to_numpy(dtype=float),
            key,
            normalizer_params,
        )
    return df


def build_training_table(ela_df, perf_df, feature_cols):
    merged = pd.merge(
        ela_df,
        perf_df,
        on=problem_key_cols(),
        how="inner",
        suffixes=("", "_perf"),
    )
    merged["problem_key"] = make_problem_key(merged)

    X_clean = clean_X(merged[feature_cols])
    for c in feature_cols:
        merged[c] = X_clean[c].values

    model_cols = list(feature_cols) + ["algname", "target_auc"]
    if INCLUDE_DIM_AS_FEATURE:
        model_cols.insert(0, "dim")
    if INCLUDE_PROBLEM_TYPE_AS_FEATURE:
        model_cols.insert(0, "problem_type")

    ag_df = merged[model_cols].copy()
    ag_df["algname"] = ag_df["algname"].astype("category")
    if INCLUDE_PROBLEM_TYPE_AS_FEATURE:
        ag_df["problem_type"] = ag_df["problem_type"].astype("category")

    keep_cols = (
        problem_key_cols()
        + ["problem_key", "problem_name", "algname", "auc_mean", "auc_transform", "transformed_auc", "target_auc"]
        + list(feature_cols)
    )
    meta_df = merged[keep_cols].copy()
    return ag_df, meta_df


def import_autogluon():
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as e:
        raise ImportError(
            "AutoGluon is not installed. Install it with:\n"
            "    pip install autogluon.tabular\n"
            "or:\n"
            "    pip install autogluon\n"
        ) from e
    return TabularPredictor


def fit_autogluon(train_ag, path, time_limit):
    """
    Fit AutoGluon only on the training fold.

    Do NOT pass the GroupKFold test fold as tuning_data when using presets
    such as "best_quality", because AutoGluon enables bagging by default and
    bagged mode does not accept external tuning_data unless use_bag_holdout=True.

    For this experiment, the test fold should remain a true external held-out
    fold for algorithm-selection evaluation.
    """
    TabularPredictor = import_autogluon()
    if os.path.exists(path):
        shutil.rmtree(path)

    predictor = TabularPredictor(
        label="target_auc",
        problem_type="regression",
        eval_metric=AG_EVAL_METRIC,
        path=path,
        verbosity=AG_VERBOSITY,
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
        ds_args=AG_DS_ARGS,
        auto_stack=AG_AUTO_STACK,
        num_bag_folds=AG_NUM_BAG_FOLDS,
        num_bag_sets=AG_NUM_BAG_SETS,
        num_stack_levels=AG_NUM_STACK_LEVELS,
        use_bag_holdout=AG_USE_BAG_HOLDOUT,
    )
    return predictor


def train_fold_fixed_baseline(train_meta):
    mean_target = train_meta.groupby("algname")["target_auc"].mean()
    return mean_target.idxmin()


def per_problem_ranking_metrics(g, pred_col="pred_target_auc"):
    spearman = g[pred_col].corr(g["target_auc"], method="spearman")
    model_top3 = set(g.sort_values(pred_col, ascending=True).head(3)["algname"].tolist())
    oracle_top3 = set(g.sort_values("target_auc", ascending=True).head(3)["algname"].tolist())
    top3_overlap = len(model_top3 & oracle_top3) / max(1, len(oracle_top3))
    return spearman, top3_overlap


def selector_validation(test_meta, pred, train_meta, fold):
    tmp = test_meta.copy()
    tmp["pred_target_auc"] = pred
    fixed_alg = train_fold_fixed_baseline(train_meta)
    rows = []

    for key, g in tmp.groupby(problem_key_cols()):
        g = g.copy()
        pred_row = g.loc[g["pred_target_auc"].idxmin()]
        oracle_row = g.loc[g["auc_mean"].idxmin()]

        fixed_rows = g[g["algname"] == fixed_alg]
        if len(fixed_rows):
            fixed_actual_auc = float(fixed_rows["auc_mean"].iloc[0])
            fixed_actual_target = float(fixed_rows["target_auc"].iloc[0])
        else:
            fixed_actual_auc = np.nan
            fixed_actual_target = np.nan

        pred_actual_auc = float(pred_row["auc_mean"])
        oracle_actual_auc = float(oracle_row["auc_mean"])
        pred_actual_target = float(pred_row["target_auc"])
        oracle_actual_target = float(oracle_row["target_auc"])
        ranking_spearman, top3_overlap = per_problem_ranking_metrics(g)

        rows.append({
            "model": "AutoGluon_Tabular",
            "fold": int(fold),
            "problem_type": key[0],
            "fid": int(key[1]),
            "iid": int(key[2]),
            "dim": int(key[3]),
            "problem_key": pred_row["problem_key"],
            "pred_selected_alg": pred_row["algname"],
            "oracle_best_alg": oracle_row["algname"],
            "fixed_baseline_alg": fixed_alg,
            "pred_selected_actual_auc": pred_actual_auc,
            "oracle_actual_auc": oracle_actual_auc,
            "fixed_baseline_actual_auc": fixed_actual_auc,
            "pred_selected_actual_target_auc": pred_actual_target,
            "oracle_actual_target_auc": oracle_actual_target,
            "fixed_baseline_actual_target_auc": fixed_actual_target,
            "raw_regret_vs_oracle": pred_actual_auc - oracle_actual_auc,
            "fixed_baseline_raw_regret_vs_oracle": fixed_actual_auc - oracle_actual_auc if np.isfinite(fixed_actual_auc) else np.nan,
            "normalized_regret_vs_oracle": pred_actual_target - oracle_actual_target,
            "fixed_baseline_normalized_regret_vs_oracle": fixed_actual_target - oracle_actual_target if np.isfinite(fixed_actual_target) else np.nan,
            "relative_regret_vs_oracle": pred_actual_auc / oracle_actual_auc - 1.0 if oracle_actual_auc > AUC_MIN_POSITIVE else np.nan,
            "fixed_baseline_relative_regret_vs_oracle": fixed_actual_auc / oracle_actual_auc - 1.0 if np.isfinite(fixed_actual_auc) and oracle_actual_auc > AUC_MIN_POSITIVE else np.nan,
            "selected_is_oracle": pred_row["algname"] == oracle_row["algname"],
            "ranking_spearman": ranking_spearman,
            "top3_overlap": top3_overlap,
            "n_algorithms_available": int(len(g)),
        })

    return pd.DataFrame(rows)


def compute_fold_metrics(test_meta, pred, selector_df, fold):
    y = test_meta["target_auc"].to_numpy(dtype=float)
    p = np.asarray(pred, dtype=float)
    rows = []

    def add(section, metric, value, source="ALL", note=""):
        rows.append({
            "model": "AutoGluon_Tabular",
            "fold": fold,
            "section": section,
            "source": source,
            "metric": metric,
            "value": value,
            "note": note,
        })

    add("regressor", "mae", float(mean_absolute_error(y, p)))
    add("regressor", "rmse", rmse(y, p))
    add("regressor", "r2", safe_r2(y, p))
    add("regressor", "spearman", safe_spearman(y, p))
    add("selector", "oracle_match_accuracy", float(selector_df["selected_is_oracle"].mean()))
    add("selector", "mean_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].mean()))
    add("selector", "median_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].median()))
    add("selector", "mean_fixed_baseline_normalized_regret_vs_oracle", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean()))
    add(
        "selector",
        "mean_normalized_regret_improvement_over_fixed_baseline",
        float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean() - selector_df["normalized_regret_vs_oracle"].mean()),
        note="positive means AutoGluon selector has lower regret than fixed baseline",
    )
    add("selector", "mean_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].mean()))
    add("selector", "mean_ranking_spearman", float(selector_df["ranking_spearman"].mean()))
    add("selector", "mean_top3_overlap", float(selector_df["top3_overlap"].mean()))

    pred_meta = test_meta.copy()
    pred_meta["pred_target_auc"] = pred
    for source, sub in pred_meta.groupby("problem_type"):
        yy = sub["target_auc"].to_numpy(dtype=float)
        pp = sub["pred_target_auc"].to_numpy(dtype=float)
        add("regressor", "mae", float(mean_absolute_error(yy, pp)), source=source)
        add("regressor", "rmse", rmse(yy, pp), source=source)
        add("regressor", "r2", safe_r2(yy, pp), source=source)
        add("regressor", "spearman", safe_spearman(yy, pp), source=source)

    for source, sub in selector_df.groupby("problem_type"):
        add("selector", "oracle_match_accuracy", float(sub["selected_is_oracle"].mean()), source=source)
        add("selector", "mean_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].mean()), source=source)
        add("selector", "mean_fixed_baseline_normalized_regret_vs_oracle", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean()), source=source)
        add(
            "selector",
            "mean_normalized_regret_improvement_over_fixed_baseline",
            float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean() - sub["normalized_regret_vs_oracle"].mean()),
            source=source,
        )
        add("selector", "mean_ranking_spearman", float(sub["ranking_spearman"].mean()), source=source)
        add("selector", "mean_top3_overlap", float(sub["top3_overlap"].mean()), source=source)

    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df):
    agg = metrics_df.groupby(["model", "section", "source", "metric"], as_index=False).agg(
        value_mean=("value", "mean"),
        value_std=("value", "std"),
        n_folds=("fold", "nunique"),
    )

    def get(section, metric, source="ALL"):
        sub = agg[(agg["section"] == section) & (agg["metric"] == metric) & (agg["source"] == source)]
        if len(sub) == 0:
            return np.nan
        return float(sub["value_mean"].iloc[0])

    leaderboard = pd.DataFrame([{
        "model": "AutoGluon_Tabular",
        "regressor_mae": get("regressor", "mae"),
        "regressor_rmse": get("regressor", "rmse"),
        "regressor_r2": get("regressor", "r2"),
        "regressor_spearman": get("regressor", "spearman"),
        "oracle_match_accuracy": get("selector", "oracle_match_accuracy"),
        "mean_normalized_regret_vs_oracle": get("selector", "mean_normalized_regret_vs_oracle"),
        "mean_fixed_baseline_normalized_regret_vs_oracle": get("selector", "mean_fixed_baseline_normalized_regret_vs_oracle"),
        "mean_normalized_regret_improvement_over_fixed_baseline": get("selector", "mean_normalized_regret_improvement_over_fixed_baseline"),
        "mean_relative_regret_vs_oracle": get("selector", "mean_relative_regret_vs_oracle"),
        "mean_ranking_spearman": get("selector", "mean_ranking_spearman"),
        "mean_top3_overlap": get("selector", "mean_top3_overlap"),
    }])
    return agg, leaderboard


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
    plt.title("AutoGluon regressor: predicted vs true")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    text = f"MAE={mean_absolute_error(y, p):.4f}\nR²={safe_r2(y, p):.4f}\nSpearman={safe_spearman(y, p):.4f}"
    plt.text(0.04, 0.96, text, transform=plt.gca().transAxes, va="top", ha="left", bbox=dict(boxstyle="round", alpha=0.15))
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "autogluon_pred_vs_true_by_source.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_residual(pred_df):
    pred_df = pred_df.copy()
    pred_df["error"] = pred_df["pred_target_auc"] - pred_df["target_auc"]
    plt.figure(figsize=(7, 5))
    for source, sub in pred_df.groupby("problem_type"):
        plt.scatter(sub["pred_target_auc"], sub["error"], s=18, alpha=0.50, label=source)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted target")
    plt.ylabel("Prediction error: pred - true")
    plt.title("AutoGluon regressor residual plot")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "autogluon_residual_by_source.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_selector_regret(selector_df):
    by_source = selector_df.groupby("problem_type", as_index=False).agg(
        model_regret=("normalized_regret_vs_oracle", "mean"),
        fixed_baseline_regret=("fixed_baseline_normalized_regret_vs_oracle", "mean"),
        oracle_match=("selected_is_oracle", "mean"),
        top3_overlap=("top3_overlap", "mean"),
    )
    by_source.to_csv(os.path.join(OUT_DIR, "autogluon_selector_metrics_by_source.csv"), index=False)

    x = np.arange(len(by_source))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, by_source["model_regret"], width=width, label="AutoGluon selector")
    plt.bar(x + width / 2, by_source["fixed_baseline_regret"], width=width, label="Fixed baseline")
    plt.xticks(x, by_source["problem_type"])
    plt.ylabel("Mean normalized regret vs oracle")
    plt.title("AutoGluon selector regret by source")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "autogluon_selector_regret_by_source.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_metric_bars(leaderboard):
    metrics = [
        ("regressor_mae", "MAE", False),
        ("regressor_spearman", "Spearman", True),
        ("oracle_match_accuracy", "Oracle match accuracy", True),
        ("mean_normalized_regret_vs_oracle", "Mean normalized regret", False),
        ("mean_top3_overlap", "Top-3 overlap", True),
    ]
    for metric, ylabel, higher in metrics:
        val = float(leaderboard[metric].iloc[0])
        plt.figure(figsize=(5, 5))
        plt.bar(["AutoGluon"], [val])
        plt.ylabel(ylabel)
        plt.title(f"AutoGluon: {metric}")
        plt.grid(axis="y", alpha=0.25)
        plt.text(0, val, f"{val:.4g}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"autogluon_{metric}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved: {path}")


def plot_autogluon_model_leaderboard(cv_leaderboards):
    if not cv_leaderboards:
        return
    df = pd.concat(cv_leaderboards, ignore_index=True, sort=False)
    df.to_csv(os.path.join(OUT_DIR, "autogluon_internal_leaderboards_all_folds.csv"), index=False)
    if "model" not in df.columns or "score_val" not in df.columns:
        return
    agg = df.groupby("model", as_index=False).agg(
        score_val_mean=("score_val", "mean"),
        score_val_std=("score_val", "std"),
        n_folds=("fold", "nunique"),
    ).sort_values("score_val_mean", ascending=False)
    agg.to_csv(os.path.join(OUT_DIR, "autogluon_internal_model_leaderboard_summary.csv"), index=False)

    plot_df = agg.head(20).sort_values("score_val_mean", ascending=True)
    plt.figure(figsize=(9, max(5, 0.32 * len(plot_df))))
    plt.barh(plot_df["model"], plot_df["score_val_mean"])
    plt.xlabel("AutoGluon validation score, higher is better")
    plt.ylabel("Internal AutoGluon model")
    plt.title("AutoGluon internal model leaderboard, mean across folds")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "autogluon_internal_model_leaderboard_top20.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def main():
    print("=== Loading and preparing data ===")
    ela_df, perf_df = load_all_data()
    normalizer_params = fit_problem_normalizers(perf_df)
    perf_df = add_problem_normalized_target(perf_df, normalizer_params)

    feature_cols = get_feature_cols(ela_df)
    if not feature_cols:
        raise RuntimeError("No numeric ELA feature columns found.")

    ag_df, meta_df = build_training_table(ela_df, perf_df, feature_cols)
    groups = make_cv_group(meta_df)

    print(f"Rows: {len(ag_df)}")
    print(f"Problems/groups for CV: {len(np.unique(groups))}")
    print(f"ELA features: {len(feature_cols)}")
    print(f"Algorithms: {meta_df['algname'].nunique()}")
    print("Rows by source:")
    print(meta_df["problem_type"].value_counts())

    ag_df.to_csv(os.path.join(OUT_DIR, "autogluon_training_table.csv"), index=False)
    meta_df.to_csv(os.path.join(OUT_DIR, "autogluon_training_meta_table.csv"), index=False)
    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(OUT_DIR, "problem_normalizer_params.json"), "w") as f:
        json.dump(normalizer_params, f, indent=2)

    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Not enough problem groups for GroupKFold.")

    cv = GroupKFold(n_splits=n_splits)
    all_pred_parts, all_selector_parts, all_metric_parts, cv_leaderboards = [], [], [], []

    print("\n=== AutoGluon GroupKFold validation ===")
    for fold, (train_idx, test_idx) in enumerate(cv.split(ag_df, ag_df["target_auc"], groups)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        start = time.time()

        train_ag = ag_df.iloc[train_idx].copy()
        test_ag = ag_df.iloc[test_idx].copy()
        train_meta = meta_df.iloc[train_idx].copy()
        test_meta = meta_df.iloc[test_idx].copy()

        predictor_path = os.path.join(PREDICTOR_DIR, f"fold_{fold}")
        predictor = fit_autogluon(train_ag, predictor_path, AG_TIME_LIMIT_PER_FOLD)

        try:
            lb = predictor.leaderboard(test_ag, silent=True)
            lb["fold"] = fold
            cv_leaderboards.append(lb)
            lb.to_csv(os.path.join(OUT_DIR, f"autogluon_internal_leaderboard_fold_{fold}.csv"), index=False)
        except Exception as e:
            print(f"[Warning] Could not produce AutoGluon leaderboard for fold {fold}: {e}")

        test_X = test_ag.drop(columns=["target_auc"])
        pred = predictor.predict(test_X).to_numpy(dtype=float)

        pred_df = test_meta.copy()
        pred_df["fold"] = fold
        pred_df["model"] = "AutoGluon_Tabular"
        pred_df["pred_target_auc"] = pred
        pred_df["error"] = pred_df["pred_target_auc"] - pred_df["target_auc"]
        pred_df["abs_error"] = np.abs(pred_df["error"])
        all_pred_parts.append(pred_df)

        selector_df = selector_validation(test_meta, pred, train_meta, fold)
        all_selector_parts.append(selector_df)

        metrics_df = compute_fold_metrics(test_meta, pred, selector_df, fold)
        metrics_df["elapsed_seconds"] = time.time() - start
        all_metric_parts.append(metrics_df)

        pred_df.to_csv(os.path.join(OUT_DIR, f"autogluon_predictions_fold_{fold}.csv"), index=False)
        selector_df.to_csv(os.path.join(OUT_DIR, f"autogluon_selector_fold_{fold}.csv"), index=False)
        metrics_df.to_csv(os.path.join(OUT_DIR, f"autogluon_metrics_fold_{fold}.csv"), index=False)
        print(f"Fold {fold} done in {time.time() - start:.1f}s.")

    pred_all = pd.concat(all_pred_parts, ignore_index=True)
    selector_all = pd.concat(all_selector_parts, ignore_index=True)
    metrics_all = pd.concat(all_metric_parts, ignore_index=True)

    pred_all.to_csv(os.path.join(OUT_DIR, "autogluon_predictions_all_folds.csv"), index=False)
    selector_all.to_csv(os.path.join(OUT_DIR, "autogluon_selector_all_folds.csv"), index=False)
    metrics_all.to_csv(os.path.join(OUT_DIR, "autogluon_metrics_all_folds_long.csv"), index=False)

    agg_metrics, leaderboard = aggregate_metrics(metrics_all)
    agg_metrics.to_csv(os.path.join(OUT_DIR, "autogluon_metrics_aggregated.csv"), index=False)
    leaderboard.to_csv(os.path.join(OUT_DIR, "autogluon_model_selection_leaderboard.csv"), index=False)

    print("\n=== AutoGluon external-CV leaderboard ===")
    print(leaderboard)

    print("\n=== Plotting ===")
    plot_pred_vs_true(pred_all)
    plot_residual(pred_all)
    plot_selector_regret(selector_all)
    plot_metric_bars(leaderboard)
    plot_autogluon_model_leaderboard(cv_leaderboards)

    if FIT_FINAL_MODEL:
        print("\n=== Fitting final AutoGluon predictor on all data ===")
        final_path = os.path.join(PREDICTOR_DIR, "final_all_data")
        TabularPredictor = import_autogluon()
        if os.path.exists(final_path):
            shutil.rmtree(final_path)

        final_predictor = TabularPredictor(
            label="target_auc",
            problem_type="regression",
            eval_metric=AG_EVAL_METRIC,
            path=final_path,
            verbosity=AG_VERBOSITY,
        )
        final_predictor.fit(
            train_data=ag_df,
            presets=AG_PRESETS,
            hyperparameters=AG_HYPERPARAMETERS,
            time_limit=AG_TIME_LIMIT_FINAL,
            num_cpus="auto",
            num_gpus="auto",
            fit_strategy=AG_FIT_STRATEGY,
            dynamic_stacking=AG_DYNAMIC_STACKING,
            ds_args=AG_DS_ARGS,
            auto_stack=AG_AUTO_STACK,
            num_bag_folds=AG_NUM_BAG_FOLDS,
            num_bag_sets=AG_NUM_BAG_SETS,
            num_stack_levels=AG_NUM_STACK_LEVELS,
            use_bag_holdout=AG_USE_BAG_HOLDOUT,
        )

        try:
            final_lb = final_predictor.leaderboard(ag_df, silent=True)
            final_lb.to_csv(os.path.join(OUT_DIR, "autogluon_final_internal_leaderboard.csv"), index=False)
        except Exception as e:
            print(f"[Warning] Could not save final AutoGluon leaderboard: {e}")
        print(f"Final AutoGluon predictor saved to: {final_path}")

    config = {
        "ag_presets": AG_PRESETS,
        "ag_time_limit_per_fold": AG_TIME_LIMIT_PER_FOLD,
        "ag_time_limit_final": AG_TIME_LIMIT_FINAL,
        "ag_eval_metric": AG_EVAL_METRIC,
        "ag_fit_strategy": AG_FIT_STRATEGY,
        "ag_dynamic_stacking": AG_DYNAMIC_STACKING,
        "ag_auto_stack": AG_AUTO_STACK,
        "ag_num_bag_folds": AG_NUM_BAG_FOLDS,
        "ag_num_bag_sets": AG_NUM_BAG_SETS,
        "ag_num_stack_levels": AG_NUM_STACK_LEVELS,
        "ag_use_bag_holdout": AG_USE_BAG_HOLDOUT,
        "ag_ds_args": AG_DS_ARGS,
        "ag_hyperparameters": AG_HYPERPARAMETERS,
        "n_splits": N_SPLITS,
        "actual_n_splits": n_splits,
        "include_problem_type_as_feature": INCLUDE_PROBLEM_TYPE_AS_FEATURE,
        "include_dim_as_feature": INCLUDE_DIM_AS_FEATURE,
        "target_transform": "BBOB/MABBOB raw AUC; LLM filtered log1p AUC; then per-problem normalization",
        "normalization_method": NORMALIZATION_METHOD,
        "cv_grouping": "BBOB grouped by fid; MABBOB/LLM grouped by problem_key; external test folds are not passed as tuning_data",
        "primary_selection_metric": "mean_normalized_regret_vs_oracle",
    }
    with open(os.path.join(OUT_DIR, "autogluon_run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
