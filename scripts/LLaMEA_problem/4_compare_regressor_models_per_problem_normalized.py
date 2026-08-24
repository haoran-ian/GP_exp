
# -*- coding: utf-8 -*-
"""
Compare multiple regressor models for ELA-based algorithm selection.

Models included:
- RandomForestRegressor
- ExtraTreesRegressor
- HistGradientBoostingRegressor
- LightGBM LGBMRegressor, if installed
- XGBoost XGBRegressor, if installed
- CatBoost CatBoostRegressor, if installed

Target:
- BBOB / MABBOB: raw AUC
- LLM: abnormal AUC filtering + log1p(AUC)
- Then per-problem normalization:
    target_auc = normalized transformed_auc within each problem instance

Main comparison metrics:
- regressor MAE / RMSE / R2 / Spearman on per-problem-normalized target
- selector oracle_match_accuracy
- selector mean_normalized_regret_vs_oracle
- selector improvement over fixed algorithm baseline

Outputs:
data/Combined/regressor_model_comparison_per_problem_normalized/
"""

# fmt: off
import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

OUT_DIR = "data/Combined/regressor_model_comparison_per_problem_normalized"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
MODEL_DIR = os.path.join(OUT_DIR, "models")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. Config
# ============================================================

RANDOM_SEED = 42
N_SPLITS = 5
SAVE_FINAL_MODELS = True
SAVE_BEST_MODEL_ALIAS = True

# AUC handling
AUC_MIN_POSITIVE = 1e-300
LLM_AUC_ABS_MAX = 1e100
LLM_AUC_UPPER_QUANTILE = 0.995
DROP_LLM_ABNORMAL_AUC = True

# Per-problem target normalization.
NORMALIZATION_METHOD = "minmax"  # "minmax", "zscore", or "robust"
NORMALIZATION_EPS = 1e-12

# To speed up first tests, set a smaller subset like ["RandomForest", "ExtraTrees"].
RUN_MODEL_NAMES = None

META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim",
    "seed", "n_samples", "instance_id", "mabbob_instance_id",
    "llm_problem_id", "selection_method", "source_dataset",
    "lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
]


# ============================================================
# 3. Data utilities
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


def ensure_problem_keys(df, problem_type):
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
                raise ValueError("MA-BBOB ELA must contain mabbob_instance_id, instance_id, or iid.")

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
        report.to_csv(os.path.join(OUT_DIR, f"auc_filter_report_{problem_type}.csv"), index=False)

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
        report.to_csv(os.path.join(OUT_DIR, f"auc_filter_report_{problem_type}.csv"), index=False)
        df.loc[abnormal].to_csv(os.path.join(OUT_DIR, f"auc_abnormal_rows_{problem_type}.csv"), index=False)

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
                raise ValueError("MA-BBOB performance must contain mabbob_instance_id, iid, or instance_id.")

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
        const_cols = nunique[nunique <= 1].index.tolist()
        feature_cols = [c for c in feature_cols if c not in const_cols]

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


def make_groups(df):
    groups = []
    for _, r in df.iterrows():
        if r["problem_type"] == "BBOB":
            groups.append(f"BBOB_F{int(r['fid'])}")
        elif r["problem_type"] == "MABBOB":
            groups.append(f"MABBOB_{int(r['iid'])}")
        elif r["problem_type"] == "LLM":
            groups.append(f"LLM_{int(r['iid'])}")
        else:
            groups.append(f"{r['problem_type']}_{int(r['fid'])}_{int(r['iid'])}")
    return np.asarray(groups)


def load_all_data():
    for p in [BBOB_ELA_PATH, BBOB_PERF_PATH, MABBOB_ELA_PATH, MABBOB_PERF_PATH, LLM_ELA_PATH, LLM_PERF_PATH]:
        require_file(p)

    bbob_ela = ensure_problem_keys(pd.read_csv(BBOB_ELA_PATH), "BBOB")
    bbob_perf = ensure_perf_keys(pd.read_csv(BBOB_PERF_PATH), "BBOB")

    mabbob_ela = ensure_problem_keys(pd.read_csv(MABBOB_ELA_PATH), "MABBOB")
    mabbob_perf = ensure_perf_keys(pd.read_csv(MABBOB_PERF_PATH), "MABBOB")

    llm_ela = ensure_problem_keys(pd.read_csv(LLM_ELA_PATH), "LLM")
    llm_perf = ensure_perf_keys(pd.read_csv(LLM_PERF_PATH), "LLM")

    ela_df = pd.concat([bbob_ela, mabbob_ela, llm_ela], ignore_index=True, sort=False)
    perf_df = pd.concat([bbob_perf, mabbob_perf, llm_perf], ignore_index=True, sort=False)

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
# 5. Training matrix
# ============================================================

def build_regressor_train_table(ela_df, perf_df):
    df = pd.merge(
        ela_df,
        perf_df,
        on=problem_key_cols(),
        how="inner",
        suffixes=("", "_perf"),
    )
    df["problem_key"] = make_problem_key(df)
    return df


def make_regressor_X(df, feature_cols, all_algorithms=None):
    X_base = clean_X(df[feature_cols])
    alg_dummies = pd.get_dummies(df["algname"].astype(str), prefix="algname")
    X = pd.concat([X_base.reset_index(drop=True), alg_dummies.reset_index(drop=True)], axis=1)

    if all_algorithms is not None:
        alg_cols = [f"algname_{a}" for a in all_algorithms]
        for c in alg_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols + alg_cols]

    return X


# ============================================================
# 6. Models
# ============================================================

def build_models():
    models = {}

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=500,
        max_features="log2",
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    models["HistGradientBoosting"] = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_SEED,
    )

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1e-3,
            reg_lambda=1e-2,
            objective="regression",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception as e:
        print(f"[Skip] LightGBM not available: {e}")

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1e-3,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception as e:
        print(f"[Skip] XGBoost not available: {e}")

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as e:
        print(f"[Skip] CatBoost not available: {e}")

    if RUN_MODEL_NAMES is not None:
        models = {k: v for k, v in models.items() if k in set(RUN_MODEL_NAMES)}

    if not models:
        raise RuntimeError("No models available to run.")

    return models


def safe_model_clone(model):
    try:
        return clone(model)
    except Exception:
        # Some external estimators may fail sklearn clone in rare versions.
        import copy
        return copy.deepcopy(model)


# ============================================================
# 7. Metrics and selector validation
# ============================================================

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


def add_metric(rows, model_name, section, metric, value, source="ALL", note=""):
    rows.append({
        "model": model_name,
        "section": section,
        "source": source,
        "metric": metric,
        "value": value,
        "note": note,
    })


def train_fold_algorithm_mean_baseline(train_rows, test_rows):
    global_mean = train_rows["target_auc"].mean()
    alg_mean = train_rows.groupby("algname")["target_auc"].mean().to_dict()
    return test_rows["algname"].map(alg_mean).fillna(global_mean).to_numpy(dtype=float)


def selector_validation_from_fold(test_rows, pred_target, train_rows, fold, model_name):
    tmp = test_rows.copy()
    tmp["pred_target_auc"] = pred_target
    rows = []

    train_alg_mean = train_rows.groupby("algname")["target_auc"].mean()
    fixed_alg = train_alg_mean.idxmin()

    for key, g in tmp.groupby(problem_key_cols()):
        g = g.copy()
        source = key[0]

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

        rows.append({
            "model": model_name,
            "fold": int(fold),
            "problem_type": source,
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
            "n_algorithms_available": int(len(g)),
        })

    return pd.DataFrame(rows)


def evaluate_model_cv(model_name, model_template, X, y, train_df, groups):
    print(f"\n=== Evaluating model: {model_name} ===")
    start = time.time()

    n_splits = min(N_SPLITS, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    pred = np.full(len(train_df), np.nan, dtype=float)
    baseline_pred = np.full(len(train_df), np.nan, dtype=float)
    fold_id = np.full(len(train_df), -1, dtype=int)
    selector_parts = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"[{model_name}] fold {fold + 1}/{n_splits}: train={len(train_idx)}, test={len(test_idx)}")
        m = safe_model_clone(model_template)

        m.fit(X.iloc[train_idx], y[train_idx])
        p = m.predict(X.iloc[test_idx])

        pred[test_idx] = p
        baseline_pred[test_idx] = train_fold_algorithm_mean_baseline(
            train_df.iloc[train_idx].copy(),
            train_df.iloc[test_idx].copy(),
        )
        fold_id[test_idx] = fold

        selector_parts.append(
            selector_validation_from_fold(
                test_rows=train_df.iloc[test_idx].copy(),
                pred_target=p,
                train_rows=train_df.iloc[train_idx].copy(),
                fold=fold,
                model_name=model_name,
            )
        )

    selector_df = pd.concat(selector_parts, ignore_index=True)

    pred_df = train_df[problem_key_cols() + ["problem_key", "problem_name", "algname", "auc_mean", "auc_transform", "transformed_auc", "target_auc"]].copy()
    pred_df["model"] = model_name
    pred_df["fold"] = fold_id
    pred_df["pred_target_auc"] = pred
    pred_df["baseline_alg_mean_pred_target_auc"] = baseline_pred
    pred_df["abs_normalized_error"] = np.abs(pred_df["target_auc"] - pred_df["pred_target_auc"])
    pred_df["baseline_abs_normalized_error"] = np.abs(pred_df["target_auc"] - pred_df["baseline_alg_mean_pred_target_auc"])

    metric_rows = []

    add_metric(metric_rows, model_name, "regressor", "r2", safe_r2(y, pred))
    add_metric(metric_rows, model_name, "regressor", "mae", float(mean_absolute_error(y, pred)))
    add_metric(metric_rows, model_name, "regressor", "rmse", rmse(y, pred))
    add_metric(metric_rows, model_name, "regressor", "spearman", safe_spearman(y, pred))
    add_metric(metric_rows, model_name, "regressor", "baseline_alg_mean_mae", float(mean_absolute_error(y, baseline_pred)))
    add_metric(metric_rows, model_name, "regressor", "mae_improvement_over_alg_mean_baseline", float(mean_absolute_error(y, baseline_pred) - mean_absolute_error(y, pred)))

    add_metric(metric_rows, model_name, "selector", "oracle_match_accuracy", float(selector_df["selected_is_oracle"].mean()))
    add_metric(metric_rows, model_name, "selector", "mean_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].mean()))
    add_metric(metric_rows, model_name, "selector", "median_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].median()))
    add_metric(metric_rows, model_name, "selector", "mean_fixed_baseline_normalized_regret_vs_oracle", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean()))
    add_metric(
        metric_rows,
        model_name,
        "selector",
        "mean_normalized_regret_improvement_over_fixed_baseline",
        float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean() - selector_df["normalized_regret_vs_oracle"].mean()),
        note="positive means model selector has lower regret than fixed baseline",
    )
    add_metric(metric_rows, model_name, "selector", "mean_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].mean()))
    add_metric(metric_rows, model_name, "selector", "median_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].median()))

    for source, sub in pred_df.groupby("problem_type"):
        yt = sub["target_auc"].to_numpy(float)
        yp = sub["pred_target_auc"].to_numpy(float)
        yb = sub["baseline_alg_mean_pred_target_auc"].to_numpy(float)

        add_metric(metric_rows, model_name, "regressor", "mae", float(mean_absolute_error(yt, yp)), source=source)
        add_metric(metric_rows, model_name, "regressor", "rmse", rmse(yt, yp), source=source)
        add_metric(metric_rows, model_name, "regressor", "r2", safe_r2(yt, yp), source=source)
        add_metric(metric_rows, model_name, "regressor", "spearman", safe_spearman(yt, yp), source=source)
        add_metric(metric_rows, model_name, "regressor", "mae_improvement_over_alg_mean_baseline", float(mean_absolute_error(yt, yb) - mean_absolute_error(yt, yp)), source=source)

    for source, sub in selector_df.groupby("problem_type"):
        add_metric(metric_rows, model_name, "selector", "oracle_match_accuracy", float(sub["selected_is_oracle"].mean()), source=source)
        add_metric(metric_rows, model_name, "selector", "mean_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].mean()), source=source)
        add_metric(metric_rows, model_name, "selector", "median_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].median()), source=source)
        add_metric(metric_rows, model_name, "selector", "mean_fixed_baseline_normalized_regret_vs_oracle", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean()), source=source)
        add_metric(
            metric_rows,
            model_name,
            "selector",
            "mean_normalized_regret_improvement_over_fixed_baseline",
            float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean() - sub["normalized_regret_vs_oracle"].mean()),
            source=source,
        )

    metrics = pd.DataFrame(metric_rows)
    elapsed = time.time() - start
    metrics["elapsed_seconds"] = elapsed

    print(f"[{model_name}] done in {elapsed:.1f}s.")
    return pred_df, selector_df, metrics


def make_leaderboard(metrics_df):
    def get_metric(model, section, metric, source="ALL"):
        sub = metrics_df[
            (metrics_df["model"] == model)
            & (metrics_df["section"] == section)
            & (metrics_df["metric"] == metric)
            & (metrics_df["source"] == source)
        ]
        if len(sub) == 0:
            return np.nan
        return float(sub["value"].iloc[0])

    rows = []
    for model in sorted(metrics_df["model"].unique()):
        rows.append({
            "model": model,
            "regressor_mae": get_metric(model, "regressor", "mae"),
            "regressor_rmse": get_metric(model, "regressor", "rmse"),
            "regressor_spearman": get_metric(model, "regressor", "spearman"),
            "regressor_mae_improvement_over_alg_mean_baseline": get_metric(model, "regressor", "mae_improvement_over_alg_mean_baseline"),
            "oracle_match_accuracy": get_metric(model, "selector", "oracle_match_accuracy"),
            "mean_normalized_regret_vs_oracle": get_metric(model, "selector", "mean_normalized_regret_vs_oracle"),
            "mean_fixed_baseline_normalized_regret_vs_oracle": get_metric(model, "selector", "mean_fixed_baseline_normalized_regret_vs_oracle"),
            "mean_normalized_regret_improvement_over_fixed_baseline": get_metric(model, "selector", "mean_normalized_regret_improvement_over_fixed_baseline"),
            "mean_relative_regret_vs_oracle": get_metric(model, "selector", "mean_relative_regret_vs_oracle"),
        })

    lb = pd.DataFrame(rows)
    # Primary sorting: lower regret, higher oracle match, lower MAE.
    lb = lb.sort_values(
        ["mean_normalized_regret_vs_oracle", "oracle_match_accuracy", "regressor_mae"],
        ascending=[True, False, True],
    )
    lb["rank_by_selector_regret"] = np.arange(1, len(lb) + 1)
    return lb


# ============================================================
# 8. Final fit and save
# ============================================================

def fit_final_model(model_name, model_template, X, y):
    m = safe_model_clone(model_template)
    m.fit(X, y)
    return m


def save_final_model_bundle(model_name, model, feature_cols, reg_feature_cols, algorithms, normalizer_params, leaderboard_row):
    bundle = {
        "regressor": model,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "reg_feature_cols": reg_feature_cols,
        "algorithms": algorithms,
        "target_transform": "mixed_raw_log_per_problem_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_level": "problem_instance",
        "problem_key_definition": problem_key_cols(),
        "problem_normalizer_params": normalizer_params,
        "auc_min_positive": AUC_MIN_POSITIVE,
        "comparison_leaderboard_row": leaderboard_row,
        "inverse_transform_note": (
            "Predictions are per-problem-normalized targets. "
            "Use predicted target directly for algorithm selection. "
            "For new unseen problems, inverse raw-AUC transform is unavailable unless a per-problem normalizer is fitted from known performances."
        ),
    }

    path = os.path.join(MODEL_DIR, f"{model_name}_per_problem_normalized_regressor.joblib")
    joblib.dump(bundle, path)
    print(f"Saved model: {path}")
    return path


# ============================================================
# 9. Plots
# ============================================================

def barplot_metric(leaderboard, metric, out_path, title, ylabel, higher_is_better=False):
    df = leaderboard.copy()
    df = df.sort_values(metric, ascending=not higher_is_better)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["model"], df[metric])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)

    for i, row in enumerate(df.itertuples()):
        val = getattr(row, metric)
        if np.isfinite(val):
            ax.text(i, val, f"{val:.4g}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def make_plots(leaderboard):
    barplot_metric(
        leaderboard,
        "mean_normalized_regret_vs_oracle",
        os.path.join(PLOT_DIR, "model_comparison_selector_regret.png"),
        "Model comparison: selector regret vs oracle",
        "Mean normalized regret vs oracle",
        higher_is_better=False,
    )

    barplot_metric(
        leaderboard,
        "oracle_match_accuracy",
        os.path.join(PLOT_DIR, "model_comparison_oracle_match_accuracy.png"),
        "Model comparison: oracle match accuracy",
        "Oracle match accuracy",
        higher_is_better=True,
    )

    barplot_metric(
        leaderboard,
        "regressor_mae",
        os.path.join(PLOT_DIR, "model_comparison_regressor_mae.png"),
        "Model comparison: normalized-target MAE",
        "MAE",
        higher_is_better=False,
    )

    barplot_metric(
        leaderboard,
        "mean_normalized_regret_improvement_over_fixed_baseline",
        os.path.join(PLOT_DIR, "model_comparison_improvement_over_fixed_baseline.png"),
        "Model comparison: improvement over fixed baseline",
        "Fixed baseline regret - model regret",
        higher_is_better=True,
    )


# ============================================================
# 10. Main
# ============================================================

def main():
    print("=== Loading data ===")
    ela_df, perf_df = load_all_data()

    normalizer_params = fit_problem_normalizers(perf_df)
    perf_df = add_problem_normalized_target(perf_df, normalizer_params)

    feature_cols = get_feature_cols(ela_df)
    if not feature_cols:
        raise RuntimeError("No numeric ELA features found.")

    train_df = build_regressor_train_table(ela_df, perf_df)
    all_algorithms = sorted(train_df["algname"].astype(str).unique().tolist())

    X = make_regressor_X(train_df, feature_cols, all_algorithms=all_algorithms)
    y = train_df["target_auc"].astype(float).to_numpy()
    groups = make_groups(train_df)

    print(f"Training rows: {len(train_df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Algorithms: {len(all_algorithms)}")
    print(f"Problem groups: {len(np.unique(groups))}")
    print("Rows by source:")
    print(train_df["problem_type"].value_counts())

    # Save base tables.
    train_df[problem_key_cols() + ["problem_key", "problem_name", "algname", "auc_mean", "auc_transform", "transformed_auc", "target_auc"]].to_csv(
        os.path.join(OUT_DIR, "training_regressor_table_keys.csv"),
        index=False,
    )

    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(os.path.join(OUT_DIR, "per_problem_normalizer_params.json"), "w") as f:
        json.dump(normalizer_params, f, indent=2)

    models = build_models()
    print("Models to run:", list(models.keys()))

    all_pred_parts = []
    all_selector_parts = []
    all_metrics_parts = []

    for model_name, model_template in models.items():
        pred_df, selector_df, metrics_df = evaluate_model_cv(
            model_name=model_name,
            model_template=model_template,
            X=X,
            y=y,
            train_df=train_df,
            groups=groups,
        )
        all_pred_parts.append(pred_df)
        all_selector_parts.append(selector_df)
        all_metrics_parts.append(metrics_df)

        pred_df.to_csv(os.path.join(OUT_DIR, f"validation_predictions_{model_name}.csv"), index=False)
        selector_df.to_csv(os.path.join(OUT_DIR, f"validation_selector_{model_name}.csv"), index=False)
        metrics_df.to_csv(os.path.join(OUT_DIR, f"validation_metrics_{model_name}.csv"), index=False)

    all_pred = pd.concat(all_pred_parts, ignore_index=True)
    all_selector = pd.concat(all_selector_parts, ignore_index=True)
    all_metrics = pd.concat(all_metrics_parts, ignore_index=True)

    all_pred.to_csv(os.path.join(OUT_DIR, "validation_predictions_all_models.csv"), index=False)
    all_selector.to_csv(os.path.join(OUT_DIR, "validation_selector_all_models.csv"), index=False)
    all_metrics.to_csv(os.path.join(OUT_DIR, "validation_metrics_all_models_long.csv"), index=False)

    leaderboard = make_leaderboard(all_metrics)
    leaderboard_path = os.path.join(OUT_DIR, "model_comparison_leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"Saved: {leaderboard_path}")
    print("\n=== Leaderboard ===")
    print(leaderboard)

    make_plots(leaderboard)

    if SAVE_FINAL_MODELS:
        print("\n=== Fitting final models on all data ===")
        saved_paths = {}
        for model_name, model_template in models.items():
            model = fit_final_model(model_name, model_template, X, y)
            row = leaderboard[leaderboard["model"] == model_name].iloc[0].to_dict()
            saved_paths[model_name] = save_final_model_bundle(
                model_name=model_name,
                model=model,
                feature_cols=feature_cols,
                reg_feature_cols=X.columns.tolist(),
                algorithms=all_algorithms,
                normalizer_params=normalizer_params,
                leaderboard_row=row,
            )

        if SAVE_BEST_MODEL_ALIAS:
            best_model_name = leaderboard.iloc[0]["model"]
            best_path = saved_paths[best_model_name]
            best_bundle = joblib.load(best_path)
            alias_path = os.path.join(MODEL_DIR, "best_per_problem_normalized_regressor.joblib")
            joblib.dump(best_bundle, alias_path)
            print(f"Saved best-model alias: {alias_path}")
            print(f"Best model by selector regret: {best_model_name}")

    config = {
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "target_transform": "mixed_raw_log_per_problem_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "auc_handling": {
            "BBOB": {"transform": "raw", "abnormal_auc_filtering": False},
            "MABBOB": {"transform": "raw", "abnormal_auc_filtering": False},
            "LLM": {
                "transform": "log1p",
                "abnormal_auc_filtering": True,
                "llm_auc_abs_max": LLM_AUC_ABS_MAX,
                "llm_auc_upper_quantile": LLM_AUC_UPPER_QUANTILE,
                "drop_llm_abnormal_auc": DROP_LLM_ABNORMAL_AUC,
            },
        },
        "models_run": list(models.keys()),
        "primary_leaderboard_sort": "mean_normalized_regret_vs_oracle ascending, oracle_match_accuracy descending, regressor_mae ascending",
    }
    with open(os.path.join(OUT_DIR, "comparison_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
