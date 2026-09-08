# -*- coding: utf-8 -*-
"""
Evaluate TabPFN 3.0 regressor for ELA-based algorithm selection.

Data: BBOB + MABBOB + LLM
Target: BBOB/MABBOB raw AUC; LLM filtered log1p(AUC); then per-problem normalization.
Validation: GroupKFold by problem group.
Outputs: data/Combined/tabpfn3_regressor_selection_per_problem_normalized/

Install:
    pip install tabpfn

If TabPFN is too slow/OOM, reduce MAX_TRAIN_ROWS_PER_FOLD or TABPFN_N_ESTIMATORS.
"""

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

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
BBOB_PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"
MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
MABBOB_PERF_PATH = "data/MABBOB/mabbob_algorithm_auc_performance.csv"
LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"
LLM_PERF_PATH = "data/LLM/llm_algorithm_auc_performance.csv"

OUT_DIR = "data/Combined/tabpfn3_regressor_selection_per_problem_normalized"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5

# TabPFN 3 settings
TABPFN_VERSION = "V3"
TABPFN_N_ESTIMATORS = 8
TABPFN_DEVICE = "auto"  # "auto", "cuda", or "cpu"
TABPFN_SOFTMAX_TEMPERATURE = 0.9
TABPFN_EXTRA_KWARGS = {}

# Set e.g. 5000 if you hit memory/runtime limits.
MAX_TRAIN_ROWS_PER_FOLD = None
MAX_TOTAL_ROWS_FOR_FINAL_FIT = None
FIT_FINAL_MODEL = True
SAVE_FINAL_MODEL = True

INCLUDE_DIM_AS_FEATURE = True
INCLUDE_PROBLEM_TYPE_AS_FEATURE = False

AUC_MIN_POSITIVE = 1e-300
LLM_AUC_ABS_MAX = 1e100
LLM_AUC_UPPER_QUANTILE = 0.995
DROP_LLM_ABNORMAL_AUC = True
NORMALIZATION_METHOD = "minmax"
NORMALIZATION_EPS = 1e-12

META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim", "seed", "n_samples",
    "instance_id", "mabbob_instance_id", "llm_problem_id", "selection_method",
    "source_dataset", "lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
]


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def harmonize_feature_names(df):
    rename = {}
    for c in df.columns:
        nc = c.replace("ela_distribution.", "ela_distr.")
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
        print(f"[Clean] {problem_type} {stage}: kept {len(df)} / {n0}.")
    return df


def ensure_ela_keys(df, problem_type):
    df = harmonize_feature_names(df.copy())
    df = drop_invalid_problem_rows(df, problem_type, "ELA")
    df["problem_type"] = problem_type
    if problem_type == "BBOB":
        df = df[df["fid"].between(1, 24)].copy()
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
        if "iid" not in df.columns:
            df["iid"] = 1
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").fillna(1).astype(int)
        df["problem_name"] = df.get("problem_name", df["fid"].apply(lambda x: f"BBOB_F{int(x)}"))
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
            valid_q = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > AUC_MIN_POSITIVE)
            if valid_q.any():
                q = df.loc[valid_q, "auc_mean"].quantile(LLM_AUC_UPPER_QUANTILE)
                if np.isfinite(q):
                    df.loc[valid_q & (df["auc_mean"] > q), "auc_filter_reason"] = f"above_q{LLM_AUC_UPPER_QUANTILE}"
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
    df = drop_invalid_problem_rows(df.copy(), problem_type, "performance")
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
            df["mabbob_instance_id"] = df["iid"] if "iid" in df.columns else df["instance_id"]
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
    paths = [BBOB_ELA_PATH, BBOB_PERF_PATH, MABBOB_ELA_PATH, MABBOB_PERF_PATH, LLM_ELA_PATH, LLM_PERF_PATH]
    for p in paths:
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
    cols = []
    for c in ela_df.columns:
        if c in excluded or c in ["FAILED", "ERROR"] or c.endswith(".FAILED") or c.endswith(".ERROR"):
            continue
        if pd.api.types.is_numeric_dtype(ela_df[c]):
            cols.append(c)
    X = ela_df[cols].replace([np.inf, -np.inf], np.nan)
    cols = [c for c in cols if not X[c].isna().all()]
    if cols:
        nunique = X[cols].nunique(dropna=True)
        cols = [c for c in cols if nunique[c] > 1]
    return sorted(cols)


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
        vals = sub["transformed_auc"].to_numpy(dtype=float)
        if NORMALIZATION_METHOD == "minmax":
            vmin = float(np.min(vals)); vmax = float(np.max(vals)); scale = vmax - vmin
            if not np.isfinite(scale) or scale <= NORMALIZATION_EPS:
                scale = 1.0
            params[key] = {"method": "minmax", "min": vmin, "max": vmax, "scale": scale}
        elif NORMALIZATION_METHOD == "zscore":
            mean = float(np.mean(vals)); std = float(np.std(vals))
            if not np.isfinite(std) or std <= NORMALIZATION_EPS:
                std = 1.0
            params[key] = {"method": "zscore", "mean": mean, "std": std}
        else:
            raise ValueError(NORMALIZATION_METHOD)
    return params


def transform_value_by_problem(value, problem_key, params):
    p = params[problem_key]
    value = np.asarray(value, dtype=float)
    if p["method"] == "minmax":
        return (value - p["min"]) / p["scale"]
    if p["method"] == "zscore":
        return (value - p["mean"]) / p["std"]
    raise ValueError(p["method"])


def add_problem_normalized_target(perf_df, params):
    df = perf_df.copy()
    df["problem_key"] = make_problem_key(df)
    df["target_auc"] = np.nan
    for key, idx in df.groupby("problem_key").groups.items():
        df.loc[idx, "target_auc"] = transform_value_by_problem(df.loc[idx, "transformed_auc"].to_numpy(float), key, params)
    return df


def build_training_table(ela_df, perf_df, feature_cols):
    merged = pd.merge(ela_df, perf_df, on=problem_key_cols(), how="inner", suffixes=("", "_perf"))
    merged["problem_key"] = make_problem_key(merged)
    X_clean = clean_X(merged[feature_cols])
    for c in feature_cols:
        merged[c] = X_clean[c].values
    alg_dummies = pd.get_dummies(merged["algname"].astype(str), prefix="algname")
    X = merged[feature_cols].copy()
    if INCLUDE_DIM_AS_FEATURE:
        X.insert(0, "dim", merged["dim"].astype(float).values)
    if INCLUDE_PROBLEM_TYPE_AS_FEATURE:
        src_dummies = pd.get_dummies(merged["problem_type"].astype(str), prefix="source")
        X = pd.concat([X.reset_index(drop=True), src_dummies.reset_index(drop=True)], axis=1)
    X = pd.concat([X.reset_index(drop=True), alg_dummies.reset_index(drop=True)], axis=1)
    X["target_auc"] = merged["target_auc"].to_numpy(dtype=float)
    meta_cols = problem_key_cols() + ["problem_key", "problem_name", "algname", "auc_mean", "auc_transform", "transformed_auc", "target_auc"]
    return X, merged[meta_cols].copy(), [c for c in X.columns if c != "target_auc"]


def limit_training_rows(train_X, train_meta, max_rows, random_state):
    if max_rows is None or len(train_X) <= max_rows:
        return train_X, train_meta
    rng = np.random.default_rng(random_state)
    keys = train_meta["problem_key"].drop_duplicates().to_numpy()
    rng.shuffle(keys)
    selected, count = [], 0
    counts = train_meta.groupby("problem_key").size().to_dict()
    for k in keys:
        n = counts[k]
        if count + n > max_rows and count > 0:
            break
        selected.append(k); count += n
        if count >= max_rows:
            break
    mask = train_meta["problem_key"].isin(selected).to_numpy()
    return train_X.loc[mask].copy(), train_meta.loc[mask].copy()


def make_tabpfn_regressor():
    try:
        from tabpfn import TabPFNRegressor, ModelVersion
    except Exception as e:
        raise ImportError("TabPFN import failed. Install with: pip install tabpfn . Python 3.10+ is recommended/required for current releases.") from e
    version = getattr(ModelVersion, TABPFN_VERSION)
    kwargs = {"n_estimators": TABPFN_N_ESTIMATORS, "softmax_temperature": TABPFN_SOFTMAX_TEMPERATURE, **TABPFN_EXTRA_KWARGS}
    if TABPFN_DEVICE != "auto":
        kwargs["device"] = TABPFN_DEVICE
    try:
        return TabPFNRegressor.create_default_for_version(version, **kwargs)
    except TypeError:
        # Fall back by progressively removing optional kwargs if installed version differs.
        keys = list(kwargs.keys())
        for k in keys:
            kwargs.pop(k, None)
            try:
                return TabPFNRegressor.create_default_for_version(version, **kwargs)
            except TypeError:
                pass
    try:
        if TABPFN_DEVICE == "auto":
            return TabPFNRegressor(n_estimators=TABPFN_N_ESTIMATORS)
        return TabPFNRegressor(n_estimators=TABPFN_N_ESTIMATORS, device=TABPFN_DEVICE)
    except Exception as e:
        raise RuntimeError("Could not construct TabPFNRegressor.") from e


def train_fold_fixed_baseline(train_meta):
    return train_meta.groupby("algname")["target_auc"].mean().idxmin()


def per_problem_ranking_metrics(g, pred_col="pred_target_auc"):
    spearman = g[pred_col].corr(g["target_auc"], method="spearman")
    model_top3 = set(g.sort_values(pred_col, ascending=True).head(3)["algname"].tolist())
    oracle_top3 = set(g.sort_values("target_auc", ascending=True).head(3)["algname"].tolist())
    top3_overlap = len(model_top3 & oracle_top3) / max(1, len(oracle_top3))
    return spearman, top3_overlap


def selector_validation(test_meta, pred, train_meta, fold):
    tmp = test_meta.copy(); tmp["pred_target_auc"] = pred
    fixed_alg = train_fold_fixed_baseline(train_meta)
    rows = []
    for key, g in tmp.groupby(problem_key_cols()):
        g = g.copy()
        pred_row = g.loc[g["pred_target_auc"].idxmin()]
        oracle_row = g.loc[g["auc_mean"].idxmin()]
        fixed_rows = g[g["algname"] == fixed_alg]
        if len(fixed_rows):
            fixed_auc = float(fixed_rows["auc_mean"].iloc[0]); fixed_target = float(fixed_rows["target_auc"].iloc[0])
        else:
            fixed_auc = np.nan; fixed_target = np.nan
        pred_auc = float(pred_row["auc_mean"]); oracle_auc = float(oracle_row["auc_mean"])
        pred_target = float(pred_row["target_auc"]); oracle_target = float(oracle_row["target_auc"])
        rank_sp, top3 = per_problem_ranking_metrics(g)
        rows.append({
            "model": "TabPFN3_Regressor", "fold": int(fold), "problem_type": key[0],
            "fid": int(key[1]), "iid": int(key[2]), "dim": int(key[3]), "problem_key": pred_row["problem_key"],
            "pred_selected_alg": pred_row["algname"], "oracle_best_alg": oracle_row["algname"], "fixed_baseline_alg": fixed_alg,
            "pred_selected_actual_auc": pred_auc, "oracle_actual_auc": oracle_auc, "fixed_baseline_actual_auc": fixed_auc,
            "pred_selected_actual_target_auc": pred_target, "oracle_actual_target_auc": oracle_target, "fixed_baseline_actual_target_auc": fixed_target,
            "raw_regret_vs_oracle": pred_auc - oracle_auc,
            "fixed_baseline_raw_regret_vs_oracle": fixed_auc - oracle_auc if np.isfinite(fixed_auc) else np.nan,
            "normalized_regret_vs_oracle": pred_target - oracle_target,
            "fixed_baseline_normalized_regret_vs_oracle": fixed_target - oracle_target if np.isfinite(fixed_target) else np.nan,
            "relative_regret_vs_oracle": pred_auc / oracle_auc - 1.0 if oracle_auc > AUC_MIN_POSITIVE else np.nan,
            "fixed_baseline_relative_regret_vs_oracle": fixed_auc / oracle_auc - 1.0 if np.isfinite(fixed_auc) and oracle_auc > AUC_MIN_POSITIVE else np.nan,
            "selected_is_oracle": pred_row["algname"] == oracle_row["algname"],
            "ranking_spearman": rank_sp, "top3_overlap": top3, "n_algorithms_available": int(len(g)),
        })
    return pd.DataFrame(rows)


def compute_fold_metrics(test_meta, pred, selector_df, fold):
    y = test_meta["target_auc"].to_numpy(float); p = np.asarray(pred, dtype=float)
    rows = []
    def add(section, metric, value, source="ALL", note=""):
        rows.append({"model": "TabPFN3_Regressor", "fold": fold, "section": section, "source": source, "metric": metric, "value": value, "note": note})
    add("regressor", "mae", float(mean_absolute_error(y, p)))
    add("regressor", "rmse", rmse(y, p))
    add("regressor", "r2", safe_r2(y, p))
    add("regressor", "spearman", safe_spearman(y, p))
    add("selector", "oracle_match_accuracy", float(selector_df["selected_is_oracle"].mean()))
    add("selector", "mean_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].mean()))
    add("selector", "median_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].median()))
    add("selector", "mean_fixed_baseline_normalized_regret_vs_oracle", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean()))
    add("selector", "mean_normalized_regret_improvement_over_fixed_baseline", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean() - selector_df["normalized_regret_vs_oracle"].mean()), note="positive means TabPFN selector has lower regret than fixed baseline")
    add("selector", "mean_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].mean()))
    add("selector", "mean_ranking_spearman", float(selector_df["ranking_spearman"].mean()))
    add("selector", "mean_top3_overlap", float(selector_df["top3_overlap"].mean()))
    pred_meta = test_meta.copy(); pred_meta["pred_target_auc"] = pred
    for src, sub in pred_meta.groupby("problem_type"):
        yy = sub["target_auc"].to_numpy(float); pp = sub["pred_target_auc"].to_numpy(float)
        add("regressor", "mae", float(mean_absolute_error(yy, pp)), source=src)
        add("regressor", "rmse", rmse(yy, pp), source=src)
        add("regressor", "r2", safe_r2(yy, pp), source=src)
        add("regressor", "spearman", safe_spearman(yy, pp), source=src)
    for src, sub in selector_df.groupby("problem_type"):
        add("selector", "oracle_match_accuracy", float(sub["selected_is_oracle"].mean()), source=src)
        add("selector", "mean_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].mean()), source=src)
        add("selector", "mean_fixed_baseline_normalized_regret_vs_oracle", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean()), source=src)
        add("selector", "mean_normalized_regret_improvement_over_fixed_baseline", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean() - sub["normalized_regret_vs_oracle"].mean()), source=src)
        add("selector", "mean_ranking_spearman", float(sub["ranking_spearman"].mean()), source=src)
        add("selector", "mean_top3_overlap", float(sub["top3_overlap"].mean()), source=src)
    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df):
    agg = metrics_df.groupby(["model", "section", "source", "metric"], as_index=False).agg(value_mean=("value", "mean"), value_std=("value", "std"), n_folds=("fold", "nunique"))
    def get(section, metric, source="ALL"):
        sub = agg[(agg["section"] == section) & (agg["metric"] == metric) & (agg["source"] == source)]
        return np.nan if len(sub) == 0 else float(sub["value_mean"].iloc[0])
    leaderboard = pd.DataFrame([{
        "model": "TabPFN3_Regressor",
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
    y = pred_df["target_auc"].to_numpy(float); p = pred_df["pred_target_auc"].to_numpy(float)
    lo = float(np.nanmin([y.min(), p.min()])); hi = float(np.nanmax([y.max(), p.max()])); pad = 0.05 * (hi - lo + 1e-12)
    plt.figure(figsize=(7, 6))
    for src, sub in pred_df.groupby("problem_type"):
        plt.scatter(sub["target_auc"], sub["pred_target_auc"], s=18, alpha=0.50, label=f"{src} (n={len(sub)})")
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    plt.xlabel("True per-problem-normalized target")
    plt.ylabel("Predicted target")
    plt.title("TabPFN 3.0 regressor: predicted vs true")
    plt.grid(alpha=0.25); plt.legend(frameon=True)
    text = f"MAE={mean_absolute_error(y, p):.4f}\nR²={safe_r2(y, p):.4f}\nSpearman={safe_spearman(y, p):.4f}"
    plt.text(0.04, 0.96, text, transform=plt.gca().transAxes, va="top", ha="left", bbox=dict(boxstyle="round", alpha=0.15))
    plt.tight_layout(); path = os.path.join(PLOT_DIR, "tabpfn3_pred_vs_true_by_source.png"); plt.savefig(path, dpi=300); plt.close(); print(f"Saved: {path}")


def plot_residual(pred_df):
    df = pred_df.copy(); df["error"] = df["pred_target_auc"] - df["target_auc"]
    plt.figure(figsize=(7, 5))
    for src, sub in df.groupby("problem_type"):
        plt.scatter(sub["pred_target_auc"], sub["error"], s=18, alpha=0.50, label=src)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted target"); plt.ylabel("Prediction error: pred - true")
    plt.title("TabPFN 3.0 regressor residual plot")
    plt.grid(alpha=0.25); plt.legend(frameon=True)
    plt.tight_layout(); path = os.path.join(PLOT_DIR, "tabpfn3_residual_by_source.png"); plt.savefig(path, dpi=300); plt.close(); print(f"Saved: {path}")


def plot_selector_regret(selector_df):
    by_src = selector_df.groupby("problem_type", as_index=False).agg(
        model_regret=("normalized_regret_vs_oracle", "mean"),
        fixed_baseline_regret=("fixed_baseline_normalized_regret_vs_oracle", "mean"),
        oracle_match=("selected_is_oracle", "mean"),
        top3_overlap=("top3_overlap", "mean"),
    )
    by_src.to_csv(os.path.join(OUT_DIR, "tabpfn3_selector_metrics_by_source.csv"), index=False)
    x = np.arange(len(by_src)); width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, by_src["model_regret"], width=width, label="TabPFN3 selector")
    plt.bar(x + width / 2, by_src["fixed_baseline_regret"], width=width, label="Fixed baseline")
    plt.xticks(x, by_src["problem_type"]); plt.ylabel("Mean normalized regret vs oracle")
    plt.title("TabPFN 3.0 selector regret by source")
    plt.grid(axis="y", alpha=0.25); plt.legend(frameon=True)
    plt.tight_layout(); path = os.path.join(PLOT_DIR, "tabpfn3_selector_regret_by_source.png"); plt.savefig(path, dpi=300); plt.close(); print(f"Saved: {path}")


def plot_metric_bars(leaderboard):
    metrics = [("regressor_mae", "MAE", False), ("regressor_spearman", "Spearman", True), ("oracle_match_accuracy", "Oracle match accuracy", True), ("mean_normalized_regret_vs_oracle", "Mean normalized regret", False), ("mean_top3_overlap", "Top-3 overlap", True)]
    for metric, ylabel, higher in metrics:
        val = float(leaderboard[metric].iloc[0])
        plt.figure(figsize=(5, 5)); plt.bar(["TabPFN3"], [val])
        plt.ylabel(ylabel); plt.title(f"TabPFN 3.0: {metric}"); plt.grid(axis="y", alpha=0.25)
        plt.text(0, val, f"{val:.4g}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout(); path = os.path.join(PLOT_DIR, f"tabpfn3_{metric}.png"); plt.savefig(path, dpi=300); plt.close(); print(f"Saved: {path}")


def plot_compare_with_existing_if_available(tabpfn_leaderboard):
    rows = []
    r = tabpfn_leaderboard.iloc[0].to_dict(); r["family"] = "TabPFN3"; rows.append(r)
    ag_path = "data/Combined/autogluon_regressor_selection_per_problem_normalized/autogluon_model_selection_leaderboard.csv"
    if os.path.exists(ag_path):
        ag = pd.read_csv(ag_path)
        if len(ag):
            r = ag.iloc[0].to_dict(); r["family"] = "AutoGluon"; rows.append(r)
    for fam, path in [("RF", "data/Combined/source_specific_rf_regressors/rf_quality_plots/rf_quality_metrics.csv"), ("MLP", "data/Combined/source_specific_mlp_regressors/mlp_quality_plots/mlp_quality_metrics.csv")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            for _, r0 in df.iterrows():
                rows.append({"model": r0["model_name"], "family": fam, "regressor_mae": r0.get("mae", np.nan), "regressor_rmse": r0.get("rmse", np.nan), "regressor_r2": r0.get("r2", np.nan), "regressor_spearman": r0.get("spearman", np.nan), "oracle_match_accuracy": np.nan, "mean_normalized_regret_vs_oracle": np.nan, "mean_top3_overlap": np.nan})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(OUT_DIR, "tabpfn3_compare_with_existing_models_if_available.csv"), index=False)
    for metric, ylabel, higher in [("regressor_mae", "MAE", False), ("regressor_spearman", "Spearman", True), ("mean_normalized_regret_vs_oracle", "Mean normalized regret", False), ("oracle_match_accuracy", "Oracle match accuracy", True)]:
        if metric not in comp.columns:
            continue
        plot_df = comp.dropna(subset=[metric]).copy()
        if plot_df.empty:
            continue
        plot_df = plot_df.sort_values(metric, ascending=not higher)
        labels = plot_df["model"].astype(str).tolist()
        plt.figure(figsize=(max(8, 0.55 * len(labels)), 5)); plt.bar(labels, plot_df[metric])
        plt.ylabel(ylabel); plt.title(f"TabPFN3 comparison if available: {metric}")
        plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.25)
        plt.tight_layout(); path = os.path.join(PLOT_DIR, f"tabpfn3_compare_{metric}_if_available.png"); plt.savefig(path, dpi=300); plt.close(); print(f"Saved: {path}")


def main():
    print("=== Loading and preparing data ===")
    ela_df, perf_df = load_all_data()
    norm_params = fit_problem_normalizers(perf_df)
    perf_df = add_problem_normalized_target(perf_df, norm_params)
    feature_cols = get_feature_cols(ela_df)
    if not feature_cols:
        raise RuntimeError("No numeric ELA feature columns found.")
    train_df, meta_df, input_cols = build_training_table(ela_df, perf_df, feature_cols)
    groups = make_cv_group(meta_df)
    print(f"Rows: {len(train_df)}")
    print(f"Problems/groups for CV: {len(np.unique(groups))}")
    print(f"ELA features: {len(feature_cols)}")
    print(f"Input features after algorithm one-hot: {len(input_cols)}")
    print(f"Algorithms: {meta_df['algname'].nunique()}")
    print("Rows by source:"); print(meta_df["problem_type"].value_counts())
    train_df.to_csv(os.path.join(OUT_DIR, "tabpfn3_training_table.csv"), index=False)
    meta_df.to_csv(os.path.join(OUT_DIR, "tabpfn3_training_meta_table.csv"), index=False)
    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f: json.dump(feature_cols, f, indent=2)
    with open(os.path.join(OUT_DIR, "feature_input_cols.json"), "w") as f: json.dump(input_cols, f, indent=2)
    with open(os.path.join(OUT_DIR, "problem_normalizer_params.json"), "w") as f: json.dump(norm_params, f, indent=2)
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2: raise RuntimeError("Not enough problem groups for GroupKFold.")
    cv = GroupKFold(n_splits=n_splits)
    all_pred, all_sel, all_met = [], [], []
    print("\n=== TabPFN 3.0 GroupKFold validation ===")
    for fold, (tr_idx, te_idx) in enumerate(cv.split(train_df, train_df["target_auc"], groups)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        start = time.time()
        tr_df = train_df.iloc[tr_idx].copy(); te_df = train_df.iloc[te_idx].copy()
        tr_meta = meta_df.iloc[tr_idx].copy(); te_meta = meta_df.iloc[te_idx].copy()
        tr_df, tr_meta = limit_training_rows(tr_df, tr_meta, MAX_TRAIN_ROWS_PER_FOLD, RANDOM_SEED + fold)
        X_train = tr_df[input_cols].copy(); y_train = tr_df["target_auc"].to_numpy(float); X_test = te_df[input_cols].copy()
        print(f"Fit rows: {len(X_train)}, test rows: {len(X_test)}")
        model = make_tabpfn_regressor(); model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test), dtype=float)
        pred_df = te_meta.copy(); pred_df["fold"] = fold; pred_df["model"] = "TabPFN3_Regressor"; pred_df["pred_target_auc"] = pred; pred_df["error"] = pred_df["pred_target_auc"] - pred_df["target_auc"]; pred_df["abs_error"] = np.abs(pred_df["error"])
        sel_df = selector_validation(te_meta, pred, tr_meta, fold)
        met_df = compute_fold_metrics(te_meta, pred, sel_df, fold); met_df["elapsed_seconds"] = time.time() - start
        all_pred.append(pred_df); all_sel.append(sel_df); all_met.append(met_df)
        pred_df.to_csv(os.path.join(OUT_DIR, f"tabpfn3_predictions_fold_{fold}.csv"), index=False)
        sel_df.to_csv(os.path.join(OUT_DIR, f"tabpfn3_selector_fold_{fold}.csv"), index=False)
        met_df.to_csv(os.path.join(OUT_DIR, f"tabpfn3_metrics_fold_{fold}.csv"), index=False)
        print(f"Fold {fold} done in {time.time() - start:.1f}s.")
    pred_all = pd.concat(all_pred, ignore_index=True); sel_all = pd.concat(all_sel, ignore_index=True); met_all = pd.concat(all_met, ignore_index=True)
    pred_all.to_csv(os.path.join(OUT_DIR, "tabpfn3_predictions_all_folds.csv"), index=False)
    sel_all.to_csv(os.path.join(OUT_DIR, "tabpfn3_selector_all_folds.csv"), index=False)
    met_all.to_csv(os.path.join(OUT_DIR, "tabpfn3_metrics_all_folds_long.csv"), index=False)
    agg, leaderboard = aggregate_metrics(met_all)
    agg.to_csv(os.path.join(OUT_DIR, "tabpfn3_metrics_aggregated.csv"), index=False)
    leaderboard.to_csv(os.path.join(OUT_DIR, "tabpfn3_model_selection_leaderboard.csv"), index=False)
    print("\n=== TabPFN 3.0 external-CV leaderboard ==="); print(leaderboard)
    plot_pred_vs_true(pred_all); plot_residual(pred_all); plot_selector_regret(sel_all); plot_metric_bars(leaderboard); plot_compare_with_existing_if_available(leaderboard)
    if FIT_FINAL_MODEL:
        print("\n=== Fitting final TabPFN 3.0 model on all data ===")
        final_df, final_meta = limit_training_rows(train_df.copy(), meta_df.copy(), MAX_TOTAL_ROWS_FOR_FINAL_FIT, RANDOM_SEED)
        X_final = final_df[input_cols].copy(); y_final = final_df["target_auc"].to_numpy(float)
        final_model = make_tabpfn_regressor(); final_model.fit(X_final, y_final)
        if SAVE_FINAL_MODEL:
            bundle = {"model": final_model, "model_name": "TabPFN3_Regressor", "tabpfn_version": TABPFN_VERSION, "tabpfn_n_estimators": TABPFN_N_ESTIMATORS, "tabpfn_device": TABPFN_DEVICE, "feature_cols": feature_cols, "feature_input_cols": input_cols, "target_transform": "BBOB/MABBOB raw AUC; LLM filtered log1p AUC; then per-problem normalization", "normalization_method": NORMALIZATION_METHOD, "problem_normalizer_params": norm_params, "problem_key_definition": problem_key_cols(), "leaderboard": leaderboard.iloc[0].to_dict(), "note": "Use predicted target_auc for algorithm selection; lower predicted target is better."}
            final_path = os.path.join(MODEL_DIR, "tabpfn3_final_regressor.joblib"); joblib.dump(bundle, final_path); print(f"Saved final model bundle: {final_path}")
    config = {"tabpfn_version": TABPFN_VERSION, "tabpfn_n_estimators": TABPFN_N_ESTIMATORS, "tabpfn_device": TABPFN_DEVICE, "tabpfn_softmax_temperature": TABPFN_SOFTMAX_TEMPERATURE, "tabpfn_extra_kwargs": TABPFN_EXTRA_KWARGS, "max_train_rows_per_fold": MAX_TRAIN_ROWS_PER_FOLD, "max_total_rows_for_final_fit": MAX_TOTAL_ROWS_FOR_FINAL_FIT, "n_splits": N_SPLITS, "actual_n_splits": n_splits, "include_dim_as_feature": INCLUDE_DIM_AS_FEATURE, "include_problem_type_as_feature": INCLUDE_PROBLEM_TYPE_AS_FEATURE, "target_transform": "BBOB/MABBOB raw AUC; LLM filtered log1p AUC; then per-problem normalization", "normalization_method": NORMALIZATION_METHOD, "cv_grouping": "BBOB grouped by fid; MABBOB/LLM grouped by problem_key", "primary_selection_metric": "mean_normalized_regret_vs_oracle"}
    with open(os.path.join(OUT_DIR, "tabpfn3_run_config.json"), "w") as f: json.dump(config, f, indent=2)
    print("\nDone."); print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
