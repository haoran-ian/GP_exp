
# -*- coding: utf-8 -*-
"""
Train source-specific regressors and run real-world ELA ablation.

Sources:
    1. BBOB-only
    2. MABBOB-only
    3. LLM-only

Models per source:
    1. RandomForestRegressor only

Then use every trained model for real-world ELA feature-set ablation.

Important:
---------
The real-world ablation part does NOT require real-world performance data.
It uses model prediction changes only, like the previous simple ablation script.

Outputs:
--------
data/Combined/source_specific_rf_mlp_regressors/
results/Combined/source_specific_rf_mlp_realworld_ablation/
"""

# fmt: off
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# fmt: on

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
BBOB_PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"

MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
MABBOB_PERF_PATH = "data/MABBOB/mabbob_algorithm_auc_performance.csv"

LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"
LLM_PERF_PATH = "data/LLM/llm_algorithm_auc_performance.csv"

REALWORLD_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"

MODEL_OUT_DIR = "data/Combined/source_specific_rf_regressors"
ABLATION_OUT_DIR = "results/Combined/source_specific_rf_realworld_ablation"

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(ABLATION_OUT_DIR, exist_ok=True)


# ============================================================
# 2. Config
# ============================================================

RANDOM_SEED = 42
N_SPLITS = 5

# AUC handling
# BBOB / MABBOB: raw AUC
# LLM: abnormal filtering + log1p(AUC)
AUC_MIN_POSITIVE = 1e-300
LLM_AUC_ABS_MAX = 1e100
LLM_AUC_UPPER_QUANTILE = 0.995
DROP_LLM_ABNORMAL_AUC = True

# Per-problem normalization inside each source
NORMALIZATION_METHOD = "minmax"
NORMALIZATION_EPS = 1e-12

# Ablation
N_REPEATS = 5
ABLATION_MODE = "uniform"  # "uniform" or "permutation"

# Optional custom feature set CSV:
# Columns can be:
#   feature_set, feature
# If None, use ELA group-level feature sets.
FEATURE_SET_CSV = None
DEFAULT_FEATURE_SETS = {}
AUTO_GROUP_FEATURE_SETS = True
INCLUDE_FULL_FEATURE_SET = True

# If running is slow, set e.g.:
# RUN_SOURCES = ["BBOB"]
# RUN_MODELS = ["RF"]
RUN_SOURCES = ["BBOB", "MABBOB", "LLM"]
RUN_MODELS = ["RF"]

META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim",
    "seed", "n_samples", "instance_id", "mabbob_instance_id",
    "llm_problem_id", "selection_method", "source_dataset",
    "lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
]


# ============================================================
# 3. General utilities
# ============================================================

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


def safe_spearman(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return np.nan
    return float(r2_score(y_true, y_pred))


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
        print(f"[Clean] {problem_type} {stage}: kept {len(df)} / {n0}.")
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

        report = (
            df.groupby(["problem_type", "auc_filter_reason"], dropna=False)
            .size()
            .reset_index(name="n_rows")
        )
        report.to_csv(os.path.join(MODEL_OUT_DIR, f"auc_filter_report_{problem_type}.csv"), index=False)

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

        report = (
            df.groupby(["problem_type", "auc_filter_reason"], dropna=False)
            .size()
            .reset_index(name="n_rows")
        )
        report.to_csv(os.path.join(MODEL_OUT_DIR, f"auc_filter_report_{problem_type}.csv"), index=False)
        df.loc[abnormal].to_csv(os.path.join(MODEL_OUT_DIR, f"auc_abnormal_rows_{problem_type}.csv"), index=False)

        if DROP_LLM_ABNORMAL_AUC:
            df = df.loc[~abnormal].copy()
            print(f"[AUC clean] LLM: dropped {n0 - len(df)} abnormal rows; kept {len(df)} / {n0}.")
        else:
            df = df.loc[~invalid].copy()
            if LLM_AUC_ABS_MAX is not None:
                df["auc_mean"] = df["auc_mean"].clip(upper=LLM_AUC_ABS_MAX)
            if LLM_AUC_UPPER_QUANTILE is not None:
                q = df["auc_mean"].quantile(LLM_AUC_UPPER_QUANTILE)
                if np.isfinite(q):
                    df["auc_mean"] = df["auc_mean"].clip(upper=q)

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
    X = X.clip(lower=-clip_abs, upper=clip_abs)

    float32_max = np.finfo(np.float32).max / 100.0
    X = X.clip(lower=-float32_max, upper=float32_max)
    return X.astype(np.float32)


def source_paths(source):
    if source == "BBOB":
        return BBOB_ELA_PATH, BBOB_PERF_PATH
    if source == "MABBOB":
        return MABBOB_ELA_PATH, MABBOB_PERF_PATH
    if source == "LLM":
        return LLM_ELA_PATH, LLM_PERF_PATH
    raise ValueError(source)


def load_source_data(source):
    ela_path, perf_path = source_paths(source)
    require_file(ela_path)
    require_file(perf_path)

    ela_df = ensure_ela_keys(pd.read_csv(ela_path), source)
    perf_df = ensure_perf_keys(pd.read_csv(perf_path), source)
    return ela_df, perf_df


# ============================================================
# 4. Per-problem normalization
# ============================================================

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

        elif NORMALIZATION_METHOD == "robust":
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            iqr = float(q75 - q25)
            if not np.isfinite(iqr) or iqr <= NORMALIZATION_EPS:
                iqr = 1.0
            params[key] = {
                "method": "robust",
                "median": median,
                "q25": float(q25),
                "q75": float(q75),
                "iqr": iqr,
            }

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
    if p["method"] == "robust":
        return (value - p["median"]) / p["iqr"]

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


# ============================================================
# 5. Training table and models
# ============================================================

def build_train_table(ela_df, perf_df):
    df = pd.merge(
        ela_df,
        perf_df,
        on=problem_key_cols(),
        how="inner",
        suffixes=("", "_perf"),
    )
    df["problem_key"] = make_problem_key(df)
    return df


def make_regressor_X(df, feature_cols, algorithms=None):
    X_base = clean_X(df[feature_cols])
    alg_dummies = pd.get_dummies(df["algname"].astype(str), prefix="algname")

    X = pd.concat([X_base.reset_index(drop=True), alg_dummies.reset_index(drop=True)], axis=1)

    if algorithms is not None:
        alg_cols = [f"algname_{a}" for a in algorithms]
        for c in alg_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols + alg_cols]

    return X


def make_model(model_type):
    if model_type == "RF":
        return RandomForestRegressor(
            n_estimators=500,
            max_features="log2",
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    raise ValueError(model_type)


def make_cv_groups(train_df):
    if train_df["problem_type"].iloc[0] == "BBOB":
        # BBOB grouped by function id, consistent with previous scripts.
        return train_df["fid"].apply(lambda x: f"BBOB_F{int(x)}").to_numpy()
    else:
        # MABBOB and LLM grouped by individual problem instance.
        return train_df["problem_key"].to_numpy()


def validate_model_cv(model, X, y, train_df):
    groups = make_cv_groups(train_df)
    n_splits = min(N_SPLITS, len(np.unique(groups)))

    if n_splits < 2:
        return {
            "cv_mae": np.nan,
            "cv_rmse": np.nan,
            "cv_r2": np.nan,
            "cv_spearman": np.nan,
            "n_splits": n_splits,
        }, None

    cv = GroupKFold(n_splits=n_splits)
    pred = np.full(len(train_df), np.nan)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])

    metrics = {
        "cv_mae": float(mean_absolute_error(y, pred)),
        "cv_rmse": rmse(y, pred),
        "cv_r2": safe_r2(y, pred),
        "cv_spearman": safe_spearman(y, pred),
        "n_splits": n_splits,
    }

    pred_df = train_df[problem_key_cols() + ["problem_key", "problem_name", "algname", "auc_mean", "transformed_auc", "target_auc"]].copy()
    pred_df["pred_target_auc"] = pred
    pred_df["abs_error"] = np.abs(pred_df["target_auc"] - pred_df["pred_target_auc"])
    return metrics, pred_df


def train_source_model(source, model_type):
    print(f"\n=== Training {source}-{model_type} ===")
    start = time.time()

    ela_df, perf_df = load_source_data(source)

    normalizer_params = fit_problem_normalizers(perf_df)
    perf_df = add_problem_normalized_target(perf_df, normalizer_params)

    feature_cols = get_feature_cols(ela_df)
    if not feature_cols:
        raise RuntimeError(f"No ELA feature columns found for {source}.")

    train_df = build_train_table(ela_df, perf_df)
    algorithms = sorted(train_df["algname"].astype(str).unique().tolist())

    X = make_regressor_X(train_df, feature_cols, algorithms=algorithms)
    y = train_df["target_auc"].astype(float).to_numpy()

    model = make_model(model_type)

    cv_metrics, cv_pred_df = validate_model_cv(model, X, y, train_df)

    model.fit(X, y)

    model_name = f"{source}_{model_type}"
    model_path = os.path.join(MODEL_OUT_DIR, f"{model_name}_regressor.joblib")

    bundle = {
        "model": model,
        "source": source,
        "model_type": model_type,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "reg_feature_cols": X.columns.tolist(),
        "algorithms": algorithms,
        "target_transform": "source_specific_raw_or_log_per_problem_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_level": "problem_instance",
        "problem_key_definition": problem_key_cols(),
        "problem_normalizer_params": normalizer_params,
        "auc_handling": {
            "BBOB": "raw AUC",
            "MABBOB": "raw AUC",
            "LLM": "abnormal filtering + log1p(AUC)",
        },
        "cv_metrics": cv_metrics,
    }

    joblib.dump(bundle, model_path)

    if cv_pred_df is not None:
        cv_pred_df.to_csv(
            os.path.join(MODEL_OUT_DIR, f"{model_name}_cv_predictions.csv"),
            index=False,
        )

    elapsed = time.time() - start

    metrics_row = {
        "source": source,
        "model_type": model_type,
        "model_name": model_name,
        "n_train_rows": len(train_df),
        "n_features": len(feature_cols),
        "n_algorithms": len(algorithms),
        "model_path": model_path,
        "elapsed_seconds": elapsed,
    }
    metrics_row.update(cv_metrics)

    print(f"Saved model: {model_path}")
    print(f"CV MAE={cv_metrics['cv_mae']:.4g}, Spearman={cv_metrics['cv_spearman']:.4g}, time={elapsed:.1f}s")
    return bundle, metrics_row


# ============================================================
# 6. Real-world ELA and feature sets
# ============================================================

def load_realworld_ela():
    df = pd.read_csv(REALWORLD_ELA_PATH)
    df = harmonize_feature_names(df)

    if "fid" not in df.columns:
        raise ValueError("Real-world ELA file must contain fid.")

    fid = pd.to_numeric(df["fid"], errors="coerce")
    df = df.loc[np.isfinite(fid) & (fid < 1)].copy()

    if df.empty:
        raise RuntimeError("No real-world problems found with fid < 1.")

    if "problem_name" not in df.columns:
        df["problem_name"] = df["fid"].astype(str)
    df["problem_name"] = df["problem_name"].fillna(df["fid"].astype(str))

    return df


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


def load_feature_sets(feature_cols):
    feature_sets = {}

    if FEATURE_SET_CSV is not None:
        fs_df = pd.read_csv(FEATURE_SET_CSV)

        set_col = None
        feat_col = None

        for c in ["feature_set", "set_name", "FeatureSet", "Set"]:
            if c in fs_df.columns:
                set_col = c
                break

        for c in ["feature", "Feature", "ela_feature"]:
            if c in fs_df.columns:
                feat_col = c
                break

        if set_col is None or feat_col is None:
            raise ValueError("FEATURE_SET_CSV must contain columns like feature_set, feature.")

        for set_name, sub in fs_df.groupby(set_col):
            feature_sets[str(set_name)] = [f for f in sub[feat_col].astype(str).tolist() if f in feature_cols]

    if DEFAULT_FEATURE_SETS:
        for name, feats in DEFAULT_FEATURE_SETS.items():
            feature_sets[name] = [f for f in feats if f in feature_cols]

    if not feature_sets and AUTO_GROUP_FEATURE_SETS:
        groups = {}
        for f in feature_cols:
            groups.setdefault(get_feature_group(f), []).append(f)
        feature_sets.update(groups)

    if INCLUDE_FULL_FEATURE_SET:
        feature_sets["FULL_FEATURES"] = list(feature_cols)

    feature_sets = {name: feats for name, feats in feature_sets.items() if len(feats) > 0}
    return feature_sets


def get_uniform_ranges(real_df, feature_cols):
    ranges = {}
    for f in feature_cols:
        if f not in real_df.columns:
            ranges[f] = (-1.0, 1.0)
            continue

        s = pd.to_numeric(real_df[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if len(s) == 0:
            ranges[f] = (-1.0, 1.0)
            continue

        lo, hi = float(s.min()), float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            ranges[f] = (-1.0, 1.0)
        else:
            ranges[f] = (lo, hi)

    return ranges


def prepare_realworld_X(prob_data, feature_cols):
    available = [f for f in feature_cols if f in prob_data.columns]
    missing = sorted(set(feature_cols) - set(available))

    if missing:
        print(f"Warning: {len(missing)} model features missing in real-world ELA; filled as 0.")

    X = prob_data[available].copy()
    X = clean_X(X)

    for f in missing:
        X[f] = 0.0

    return X[feature_cols]


def make_model_input(X_ela, alg, algorithms, reg_feature_cols):
    X = X_ela.copy()

    for a in algorithms:
        X[f"algname_{a}"] = 1.0 if a == alg else 0.0

    for c in reg_feature_cols:
        if c not in X.columns:
            X[c] = 0.0

    return X[reg_feature_cols]


def predict_all_algorithms(bundle, X_ela):
    model = bundle["model"]
    algorithms = bundle["algorithms"]
    reg_feature_cols = bundle["reg_feature_cols"]

    rows = []
    for alg in algorithms:
        X_input = make_model_input(X_ela, alg, algorithms, reg_feature_cols)
        pred = model.predict(X_input)
        rows.append({
            "algname": alg,
            "pred_mean": float(np.mean(pred)),
            "pred_median": float(np.median(pred)),
        })

    return pd.DataFrame(rows)


def ablate_feature_set(X_base, features, rng, uniform_ranges):
    X_ab = X_base.copy()

    for f in features:
        if f not in X_ab.columns:
            continue

        if ABLATION_MODE == "permutation":
            vals = X_ab[f].to_numpy(copy=True)
            rng.shuffle(vals)
            X_ab[f] = vals

        elif ABLATION_MODE == "uniform":
            lo, hi = uniform_ranges.get(f, (-1.0, 1.0))
            X_ab[f] = rng.uniform(lo, hi, size=len(X_ab))

        else:
            raise ValueError(f"Unknown ABLATION_MODE: {ABLATION_MODE}")

    return X_ab


def run_ablation_for_model(bundle, real_df):
    rng = np.random.default_rng(RANDOM_SEED)

    model_name = bundle["model_name"]
    source = bundle["source"]
    model_type = bundle["model_type"]
    feature_cols = bundle["feature_cols"]

    feature_sets = load_feature_sets(feature_cols)
    uniform_ranges = get_uniform_ranges(real_df, feature_cols)

    # Save feature sets used for this model.
    fs_rows = []
    for set_name, feats in feature_sets.items():
        for f in feats:
            fs_rows.append({
                "model_name": model_name,
                "feature_set": set_name,
                "feature": f,
                "group": get_feature_group(f),
            })
    pd.DataFrame(fs_rows).to_csv(
        os.path.join(ABLATION_OUT_DIR, f"{model_name}_feature_sets_used.csv"),
        index=False,
    )

    all_problem_rows = []

    for problem_name in real_df["problem_name"].dropna().unique():
        print(f"[Ablation] {model_name} on {problem_name}")

        prob_data = real_df[real_df["problem_name"] == problem_name].copy()
        X_base = prepare_realworld_X(prob_data, feature_cols)

        base_pred_df = predict_all_algorithms(bundle, X_base)
        base_mean_all = float(base_pred_df["pred_mean"].mean())
        base_best = base_pred_df.loc[base_pred_df["pred_mean"].idxmin()]
        base_best_alg = base_best["algname"]
        base_best_pred = float(base_best["pred_mean"])

        base_pred_df.to_csv(
            os.path.join(ABLATION_OUT_DIR, f"{model_name}_{problem_name}_baseline_predictions.csv"),
            index=False,
        )

        rows = []

        for set_name, feats in feature_sets.items():
            impacts_mean_all = []
            impacts_best = []
            best_changed = []

            for rep in range(N_REPEATS):
                X_ab = ablate_feature_set(X_base, feats, rng, uniform_ranges)
                ab_pred_df = predict_all_algorithms(bundle, X_ab)

                ab_mean_all = float(ab_pred_df["pred_mean"].mean())
                ab_best = ab_pred_df.loc[ab_pred_df["pred_mean"].idxmin()]
                ab_best_alg = ab_best["algname"]
                ab_best_pred = float(ab_best["pred_mean"])

                impacts_mean_all.append(ab_mean_all - base_mean_all)
                impacts_best.append(ab_best_pred - base_best_pred)
                best_changed.append(ab_best_alg != base_best_alg)

            rows.append({
                "model_name": model_name,
                "source": source,
                "model_type": model_type,
                "problem_name": problem_name,
                "feature_set": set_name,
                "n_features": len(feats),
                "feature_groups": ";".join(sorted(set(get_feature_group(f) for f in feats))),
                "baseline_mean_pred_all_algs": base_mean_all,
                "impact_mean_pred_all_algs_mean": float(np.mean(impacts_mean_all)),
                "impact_mean_pred_all_algs_std": float(np.std(impacts_mean_all)),
                "baseline_best_alg": base_best_alg,
                "baseline_best_pred": base_best_pred,
                "impact_best_pred_mean": float(np.mean(impacts_best)),
                "impact_best_pred_std": float(np.std(impacts_best)),
                "best_alg_change_rate": float(np.mean(best_changed)),
                "n_repeats": N_REPEATS,
                "ablation_mode": ABLATION_MODE,
            })

        res = pd.DataFrame(rows).sort_values("impact_mean_pred_all_algs_mean", ascending=False)
        res.to_csv(
            os.path.join(ABLATION_OUT_DIR, f"{model_name}_{problem_name}_feature_set_ablation.csv"),
            index=False,
        )

        # Per-problem plot
        plot_df = res.sort_values("impact_mean_pred_all_algs_mean", ascending=True)
        plt.figure(figsize=(10, max(5, 0.45 * len(plot_df))))
        plt.barh(plot_df["feature_set"], plot_df["impact_mean_pred_all_algs_mean"])
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Change in mean predicted target after ablation")
        plt.ylabel("Feature set")
        plt.title(f"{model_name} | {problem_name} | ELA feature-set ablation")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(
            os.path.join(ABLATION_OUT_DIR, f"{model_name}_{problem_name}_feature_set_ablation_impact.png"),
            dpi=300,
        )
        plt.close()

        all_problem_rows.append(res)

    if not all_problem_rows:
        return pd.DataFrame()

    all_df = pd.concat(all_problem_rows, ignore_index=True)
    all_df.to_csv(
        os.path.join(ABLATION_OUT_DIR, f"{model_name}_all_realworld_ablation.csv"),
        index=False,
    )

    overall = (
        all_df.groupby(["model_name", "source", "model_type", "feature_set"], as_index=False)
        .agg(
            n_features=("n_features", "first"),
            impact_mean_pred_all_algs_mean=("impact_mean_pred_all_algs_mean", "mean"),
            impact_mean_pred_all_algs_std=("impact_mean_pred_all_algs_mean", "std"),
            impact_best_pred_mean=("impact_best_pred_mean", "mean"),
            impact_best_pred_std=("impact_best_pred_mean", "std"),
            best_alg_change_rate_mean=("best_alg_change_rate", "mean"),
            n_realworld_problems=("problem_name", "nunique"),
        )
        .sort_values("impact_mean_pred_all_algs_mean", ascending=False)
    )

    overall.to_csv(
        os.path.join(ABLATION_OUT_DIR, f"{model_name}_overall_ablation.csv"),
        index=False,
    )

    plot_df = overall.sort_values("impact_mean_pred_all_algs_mean", ascending=True)
    plt.figure(figsize=(10, max(5, 0.45 * len(plot_df))))
    plt.barh(plot_df["feature_set"], plot_df["impact_mean_pred_all_algs_mean"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Mean change in predicted target across real-world problems")
    plt.ylabel("Feature set")
    plt.title(f"{model_name} | Overall real-world ELA ablation")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(ABLATION_OUT_DIR, f"{model_name}_overall_ablation_impact.png"),
        dpi=300,
    )
    plt.close()

    return all_df


def plot_global_ablation_summary(all_ablation_df):
    if all_ablation_df.empty:
        return

    overall = (
        all_ablation_df
        .groupby(["source", "model_type", "model_name", "feature_set"], as_index=False)
        .agg(
            impact_mean_pred_all_algs_mean=("impact_mean_pred_all_algs_mean", "mean"),
            best_alg_change_rate_mean=("best_alg_change_rate", "mean"),
            n_realworld_problems=("problem_name", "nunique"),
        )
    )

    overall.to_csv(
        os.path.join(ABLATION_OUT_DIR, "overall_all_models_feature_set_ablation.csv"),
        index=False,
    )

    # One plot per feature set may be too many, so plot top feature sets by average impact.
    top_sets = (
        overall.groupby("feature_set")["impact_mean_pred_all_algs_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .index
        .tolist()
    )
    plot_df = overall[overall["feature_set"].isin(top_sets)].copy()
    plot_df["model_label"] = plot_df["source"] + "-" + plot_df["model_type"]

    for fs in top_sets:
        sub = plot_df[plot_df["feature_set"] == fs].sort_values("model_label")
        plt.figure(figsize=(10, 5))
        plt.bar(sub["model_label"], sub["impact_mean_pred_all_algs_mean"])
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.ylabel("Mean prediction change after ablation")
        plt.title(f"Feature set: {fs} | impact across source-specific models")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        safe_fs = str(fs).replace("/", "_").replace(" ", "_")
        plt.savefig(
            os.path.join(ABLATION_OUT_DIR, f"compare_models_feature_set_{safe_fs}.png"),
            dpi=300,
        )
        plt.close()


# ============================================================
# 7. Main
# ============================================================

def main():
    print("=== Train source-specific Random Forest regressors ===")

    trained_bundles = []
    train_metric_rows = []

    for source in RUN_SOURCES:
        for model_type in RUN_MODELS:
            bundle, metrics = train_source_model(source, model_type)
            trained_bundles.append(bundle)
            train_metric_rows.append(metrics)

    train_metrics = pd.DataFrame(train_metric_rows)
    train_metrics.to_csv(os.path.join(MODEL_OUT_DIR, "source_specific_model_training_summary.csv"), index=False)
    print(f"Saved: {os.path.join(MODEL_OUT_DIR, 'source_specific_model_training_summary.csv')}")

    # Plot CV MAE and Spearman
    if not train_metrics.empty:
        train_metrics["model_label"] = train_metrics["source"] + "-" + train_metrics["model_type"]

        plt.figure(figsize=(9, 5))
        plt.bar(train_metrics["model_label"], train_metrics["cv_mae"])
        plt.ylabel("CV MAE on per-problem-normalized target")
        plt.title("Source-specific model validation MAE")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_OUT_DIR, "source_specific_model_cv_mae.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.bar(train_metrics["model_label"], train_metrics["cv_spearman"])
        plt.ylabel("CV Spearman")
        plt.title("Source-specific model validation Spearman")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_OUT_DIR, "source_specific_model_cv_spearman.png"), dpi=300)
        plt.close()

    print("\n=== Real-world ELA feature-set ablation ===")
    real_df = load_realworld_ela()

    all_ablation_parts = []

    for bundle in trained_bundles:
        ab_df = run_ablation_for_model(bundle, real_df)
        if not ab_df.empty:
            all_ablation_parts.append(ab_df)

    if all_ablation_parts:
        all_ablation = pd.concat(all_ablation_parts, ignore_index=True)
        all_ablation.to_csv(
            os.path.join(ABLATION_OUT_DIR, "all_models_all_realworld_feature_set_ablation.csv"),
            index=False,
        )
        plot_global_ablation_summary(all_ablation)

    config = {
        "run_sources": RUN_SOURCES,
        "run_models": RUN_MODELS,
        "normalization_method": NORMALIZATION_METHOD,
        "target_transform": "BBOB/MABBOB raw AUC; LLM log1p filtered AUC; per-problem normalization",
        "ablation_mode": ABLATION_MODE,
        "n_repeats": N_REPEATS,
        "feature_set_csv": FEATURE_SET_CSV,
        "auto_group_feature_sets": AUTO_GROUP_FEATURE_SETS,
        "include_full_feature_set": INCLUDE_FULL_FEATURE_SET,
        "model_out_dir": MODEL_OUT_DIR,
        "ablation_out_dir": ABLATION_OUT_DIR,
    }
    with open(os.path.join(ABLATION_OUT_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Models saved to: {MODEL_OUT_DIR}")
    print(f"Ablation results saved to: {ABLATION_OUT_DIR}")


if __name__ == "__main__":
    main()
