
# -*- coding: utf-8 -*-
"""
Compare pairwise ranking models for ELA-based algorithm selection.

Pairwise formulation
--------------------
Instead of learning:

    ELA(problem) + algorithm -> target_auc

this script learns:

    ELA(problem) + algorithm_i + algorithm_j -> whether algorithm_i beats algorithm_j

where "beats" means lower target_auc on the same problem.

Prediction
----------
For each test problem, the model predicts all algorithm pairs.
Each algorithm receives a soft vote:

    score_i += P(algorithm_i beats algorithm_j)
    score_j += 1 - P(algorithm_i beats algorithm_j)

The algorithm with the highest total score is selected.

Models included
---------------
- PairwiseRandomForest
- PairwiseExtraTrees
- PairwiseHistGradientBoosting
- PairwiseLightGBMClassifier, if installed
- PairwiseXGBoostClassifier, if installed
- PairwiseCatBoostClassifier, if installed

Outputs
-------
data/Combined/pairwise_rank_model_comparison_per_problem_normalized/
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
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

OUT_DIR = "data/Combined/pairwise_rank_model_comparison_per_problem_normalized"
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

# AUC handling, same as previous regressor scripts.
AUC_MIN_POSITIVE = 1e-300
LLM_AUC_ABS_MAX = 1e100
LLM_AUC_UPPER_QUANTILE = 0.995
DROP_LLM_ABNORMAL_AUC = True

# Per-problem target normalization.
NORMALIZATION_METHOD = "minmax"
NORMALIZATION_EPS = 1e-12

# Pairwise data size control.
# None means use all algorithm pairs inside each problem.
# If you have many algorithms and it is too slow, set e.g. MAX_PAIRS_PER_PROBLEM = 300.
MAX_PAIRS_PER_PROBLEM = None

# Add both directions:
#   alg_i vs alg_j label 1
#   alg_j vs alg_i label 0
# This helps balance classes and makes the classifier symmetric in practice.
ADD_REVERSED_PAIRS = True

# To speed up first tests, set e.g. ["PairwiseExtraTrees", "PairwiseLightGBM"].
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


def make_groups_from_problem_alg_df(df):
    return df["problem_key"].to_numpy()


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
# 5. Problem-algorithm table
# ============================================================

def build_problem_alg_table(ela_df, perf_df, feature_cols):
    merged = pd.merge(
        ela_df,
        perf_df,
        on=problem_key_cols(),
        how="inner",
        suffixes=("", "_perf"),
    )
    merged["problem_key"] = make_problem_key(merged)

    # Clean ELA features first, then aggregate.
    X_clean = clean_X(merged[feature_cols])
    for c in feature_cols:
        merged[c] = X_clean[c].values

    agg_spec = {
        "problem_name": "first",
        "auc_mean": "mean",
        "auc_transform": "first",
        "transformed_auc": "mean",
        "target_auc": "mean",
    }
    for c in feature_cols:
        agg_spec[c] = "mean"

    problem_alg_df = (
        merged
        .groupby(problem_key_cols() + ["problem_key", "algname"], as_index=False)
        .agg(agg_spec)
    )

    return problem_alg_df


def make_pair_feature_columns(feature_cols, algorithms):
    cols = list(feature_cols)
    cols += [f"alg_i_{a}" for a in algorithms]
    cols += [f"alg_j_{a}" for a in algorithms]
    return cols


def make_pair_row(base_features, alg_i, alg_j, algorithms):
    row = dict(base_features)
    for a in algorithms:
        row[f"alg_i_{a}"] = 1.0 if a == alg_i else 0.0
    for a in algorithms:
        row[f"alg_j_{a}"] = 1.0 if a == alg_j else 0.0
    return row


def build_pairwise_dataset(problem_alg_df, feature_cols, algorithms, max_pairs_per_problem=None, add_reversed=True, random_state=42):
    rng = np.random.default_rng(random_state)

    rows = []
    labels = []
    groups = []
    meta_rows = []

    for pkey, g in problem_alg_df.groupby("problem_key"):
        g = g.dropna(subset=["target_auc"]).copy()
        if len(g) < 2:
            continue

        # One ELA vector per problem. All rows of same problem should have same ELA,
        # but use mean for safety.
        base_features = {f: float(g[f].mean()) for f in feature_cols}

        pairs = []
        alg_rows = list(g.itertuples(index=False))
        for i in range(len(alg_rows)):
            for j in range(i + 1, len(alg_rows)):
                ri = alg_rows[i]
                rj = alg_rows[j]

                yi = float(getattr(ri, "target_auc"))
                yj = float(getattr(rj, "target_auc"))

                if not np.isfinite(yi) or not np.isfinite(yj):
                    continue
                if yi == yj:
                    continue

                pairs.append((ri, rj))

        if max_pairs_per_problem is not None and len(pairs) > max_pairs_per_problem:
            idx = rng.choice(len(pairs), size=max_pairs_per_problem, replace=False)
            pairs = [pairs[k] for k in idx]

        for ri, rj in pairs:
            ai = getattr(ri, "algname")
            aj = getattr(rj, "algname")
            yi = float(getattr(ri, "target_auc"))
            yj = float(getattr(rj, "target_auc"))

            label = 1 if yi < yj else 0
            rows.append(make_pair_row(base_features, ai, aj, algorithms))
            labels.append(label)
            groups.append(pkey)
            meta_rows.append({
                "problem_key": pkey,
                "problem_type": getattr(ri, "problem_type"),
                "fid": getattr(ri, "fid"),
                "iid": getattr(ri, "iid"),
                "dim": getattr(ri, "dim"),
                "alg_i": ai,
                "alg_j": aj,
                "target_i": yi,
                "target_j": yj,
                "label_i_beats_j": label,
            })

            if add_reversed:
                rows.append(make_pair_row(base_features, aj, ai, algorithms))
                labels.append(1 - label)
                groups.append(pkey)
                meta_rows.append({
                    "problem_key": pkey,
                    "problem_type": getattr(ri, "problem_type"),
                    "fid": getattr(ri, "fid"),
                    "iid": getattr(ri, "iid"),
                    "dim": getattr(ri, "dim"),
                    "alg_i": aj,
                    "alg_j": ai,
                    "target_i": yj,
                    "target_j": yi,
                    "label_i_beats_j": 1 - label,
                })

    X_pair = pd.DataFrame(rows)
    y_pair = np.asarray(labels, dtype=int)
    pair_groups = np.asarray(groups)
    meta_df = pd.DataFrame(meta_rows)

    pair_cols = make_pair_feature_columns(feature_cols, algorithms)
    for c in pair_cols:
        if c not in X_pair.columns:
            X_pair[c] = 0.0
    X_pair = X_pair[pair_cols].astype(np.float32)

    return X_pair, y_pair, pair_groups, meta_df


# ============================================================
# 6. Models
# ============================================================

def build_pairwise_models():
    models = {}

    models["PairwiseRandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_features="log2",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    models["PairwiseExtraTrees"] = ExtraTreesClassifier(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    models["PairwiseHistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_SEED,
    )

    try:
        from lightgbm import LGBMClassifier
        models["PairwiseLightGBM"] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1e-3,
            reg_lambda=1e-2,
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception as e:
        print(f"[Skip] LightGBM not available: {e}")

    try:
        from xgboost import XGBClassifier
        models["PairwiseXGBoost"] = XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1e-3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception as e:
        print(f"[Skip] XGBoost not available: {e}")

    try:
        from catboost import CatBoostClassifier
        models["PairwiseCatBoost"] = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as e:
        print(f"[Skip] CatBoost not available: {e}")

    if RUN_MODEL_NAMES is not None:
        models = {k: v for k, v in models.items() if k in set(RUN_MODEL_NAMES)}

    if not models:
        raise RuntimeError("No pairwise models available.")

    return models


def safe_model_clone(model):
    try:
        return clone(model)
    except Exception:
        import copy
        return copy.deepcopy(model)


def predict_pair_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        # rare single-class fallback
        return np.full(len(X), float(model.classes_[0] == 1))
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


# ============================================================
# 7. Prediction and evaluation
# ============================================================

def predict_best_algorithm_pairwise(model, problem_rows, feature_cols, algorithms, pair_cols):
    g = problem_rows.copy()
    g = g.dropna(subset=["target_auc"]).copy()

    available_algs = g["algname"].astype(str).tolist()
    if len(available_algs) == 0:
        return None, pd.DataFrame()

    if len(available_algs) == 1:
        only_alg = available_algs[0]
        score_df = pd.DataFrame([{
            "algname": only_alg,
            "pairwise_score": 0.0,
            "n_pairwise_comparisons": 0,
        }])
        return only_alg, score_df

    base_features = {f: float(g[f].mean()) for f in feature_cols}

    rows = []
    pair_meta = []

    for i in range(len(available_algs)):
        for j in range(i + 1, len(available_algs)):
            ai = available_algs[i]
            aj = available_algs[j]
            rows.append(make_pair_row(base_features, ai, aj, algorithms))
            pair_meta.append((ai, aj))

    X_pair = pd.DataFrame(rows)
    for c in pair_cols:
        if c not in X_pair.columns:
            X_pair[c] = 0.0
    X_pair = X_pair[pair_cols].astype(np.float32)

    p_i_beats_j = predict_pair_proba(model, X_pair)

    scores = {a: 0.0 for a in available_algs}
    counts = {a: 0 for a in available_algs}

    for (ai, aj), p in zip(pair_meta, p_i_beats_j):
        p = float(p)
        scores[ai] += p
        scores[aj] += 1.0 - p
        counts[ai] += 1
        counts[aj] += 1

    score_df = pd.DataFrame([
        {
            "algname": a,
            "pairwise_score": scores[a],
            "pairwise_score_mean": scores[a] / counts[a] if counts[a] > 0 else np.nan,
            "n_pairwise_comparisons": counts[a],
        }
        for a in available_algs
    ]).sort_values("pairwise_score", ascending=False)

    best_alg = score_df.iloc[0]["algname"]
    return best_alg, score_df


def train_fold_fixed_baseline(train_problem_alg_df):
    mean_target = train_problem_alg_df.groupby("algname")["target_auc"].mean()
    return mean_target.idxmin()


def evaluate_selector_on_test(model, test_problem_alg_df, train_problem_alg_df, feature_cols, algorithms, pair_cols, fold, model_name):
    rows = []
    score_parts = []

    fixed_alg = train_fold_fixed_baseline(train_problem_alg_df)

    for pkey, g in test_problem_alg_df.groupby("problem_key"):
        best_alg, score_df = predict_best_algorithm_pairwise(
            model=model,
            problem_rows=g,
            feature_cols=feature_cols,
            algorithms=algorithms,
            pair_cols=pair_cols,
        )
        if best_alg is None:
            continue

        source = g["problem_type"].iloc[0]

        oracle_row = g.loc[g["auc_mean"].idxmin()]
        oracle_alg = oracle_row["algname"]

        selected_row = g[g["algname"] == best_alg].iloc[0]

        fixed_rows = g[g["algname"] == fixed_alg]
        if len(fixed_rows):
            fixed_row = fixed_rows.iloc[0]
            fixed_auc = float(fixed_row["auc_mean"])
            fixed_target = float(fixed_row["target_auc"])
        else:
            fixed_auc = np.nan
            fixed_target = np.nan

        score_df = score_df.copy()
        score_df["model"] = model_name
        score_df["fold"] = fold
        score_df["problem_key"] = pkey
        score_df["problem_type"] = source
        score_parts.append(score_df)

        selected_auc = float(selected_row["auc_mean"])
        oracle_auc = float(oracle_row["auc_mean"])
        selected_target = float(selected_row["target_auc"])
        oracle_target = float(oracle_row["target_auc"])

        # Ranking Spearman between model pairwise scores and true target.
        tmp_rank = pd.merge(
            score_df[["algname", "pairwise_score"]],
            g[["algname", "target_auc"]],
            on="algname",
            how="inner",
        )
        # lower true target is better, higher pairwise score is better.
        ranking_spearman = tmp_rank["pairwise_score"].corr(-tmp_rank["target_auc"], method="spearman")

        model_top3 = set(score_df.sort_values("pairwise_score", ascending=False).head(3)["algname"].tolist())
        oracle_top3 = set(g.sort_values("target_auc", ascending=True).head(3)["algname"].tolist())
        top3_overlap = len(model_top3 & oracle_top3) / max(1, len(oracle_top3))

        rows.append({
            "model": model_name,
            "fold": int(fold),
            "problem_type": source,
            "fid": int(g["fid"].iloc[0]),
            "iid": int(g["iid"].iloc[0]),
            "dim": int(g["dim"].iloc[0]),
            "problem_key": pkey,
            "pred_selected_alg": best_alg,
            "oracle_best_alg": oracle_alg,
            "fixed_baseline_alg": fixed_alg,
            "pred_selected_actual_auc": selected_auc,
            "oracle_actual_auc": oracle_auc,
            "fixed_baseline_actual_auc": fixed_auc,
            "pred_selected_actual_target_auc": selected_target,
            "oracle_actual_target_auc": oracle_target,
            "fixed_baseline_actual_target_auc": fixed_target,
            "raw_regret_vs_oracle": selected_auc - oracle_auc,
            "fixed_baseline_raw_regret_vs_oracle": fixed_auc - oracle_auc if np.isfinite(fixed_auc) else np.nan,
            "normalized_regret_vs_oracle": selected_target - oracle_target,
            "fixed_baseline_normalized_regret_vs_oracle": fixed_target - oracle_target if np.isfinite(fixed_target) else np.nan,
            "relative_regret_vs_oracle": selected_auc / oracle_auc - 1.0 if oracle_auc > AUC_MIN_POSITIVE else np.nan,
            "fixed_baseline_relative_regret_vs_oracle": fixed_auc / oracle_auc - 1.0 if np.isfinite(fixed_auc) and oracle_auc > AUC_MIN_POSITIVE else np.nan,
            "selected_is_oracle": best_alg == oracle_alg,
            "ranking_spearman": ranking_spearman,
            "top3_overlap": top3_overlap,
            "n_algorithms_available": int(len(g)),
        })

    selector_df = pd.DataFrame(rows)
    score_df_all = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    return selector_df, score_df_all


def evaluate_pair_classifier_on_test(model, X_pair_test, y_pair_test):
    pred_label = model.predict(X_pair_test)
    out = {
        "pair_accuracy": float(accuracy_score(y_pair_test, pred_label)),
    }

    if len(np.unique(y_pair_test)) == 2:
        try:
            p = predict_pair_proba(model, X_pair_test)
            out["pair_roc_auc"] = float(roc_auc_score(y_pair_test, p))
            out["pair_log_loss"] = float(log_loss(y_pair_test, np.clip(p, 1e-15, 1 - 1e-15)))
        except Exception:
            out["pair_roc_auc"] = np.nan
            out["pair_log_loss"] = np.nan
    else:
        out["pair_roc_auc"] = np.nan
        out["pair_log_loss"] = np.nan

    return out


def add_metric(rows, model_name, section, metric, value, source="ALL", note=""):
    rows.append({
        "model": model_name,
        "section": section,
        "source": source,
        "metric": metric,
        "value": value,
        "note": note,
    })


def evaluate_model_cv(model_name, model_template, problem_alg_df, feature_cols, algorithms):
    print(f"\n=== Evaluating pairwise model: {model_name} ===")
    start = time.time()

    problem_groups = problem_alg_df.drop_duplicates("problem_key")["problem_key"].to_numpy()
    n_splits = min(N_SPLITS, len(problem_groups))
    if n_splits < 2:
        raise RuntimeError("Not enough problem groups for GroupKFold.")

    # Split on unique problem keys.
    problem_key_df = problem_alg_df.drop_duplicates("problem_key")[["problem_key"]].copy()
    cv = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((len(problem_key_df), 1))
    dummy_y = np.zeros(len(problem_key_df))
    groups = problem_key_df["problem_key"].to_numpy()

    pair_cols = make_pair_feature_columns(feature_cols, algorithms)

    selector_parts = []
    score_parts = []
    metric_rows = []
    pair_metric_rows = []

    for fold, (train_problem_idx, test_problem_idx) in enumerate(cv.split(dummy_X, dummy_y, groups)):
        train_keys = set(problem_key_df.iloc[train_problem_idx]["problem_key"])
        test_keys = set(problem_key_df.iloc[test_problem_idx]["problem_key"])

        train_problem_alg = problem_alg_df[problem_alg_df["problem_key"].isin(train_keys)].copy()
        test_problem_alg = problem_alg_df[problem_alg_df["problem_key"].isin(test_keys)].copy()

        print(
            f"[{model_name}] fold {fold + 1}/{n_splits}: "
            f"train problems={len(train_keys)}, test problems={len(test_keys)}"
        )

        X_pair_train, y_pair_train, pair_groups_train, meta_train = build_pairwise_dataset(
            train_problem_alg,
            feature_cols=feature_cols,
            algorithms=algorithms,
            max_pairs_per_problem=MAX_PAIRS_PER_PROBLEM,
            add_reversed=ADD_REVERSED_PAIRS,
            random_state=RANDOM_SEED + fold,
        )

        X_pair_test, y_pair_test, pair_groups_test, meta_test = build_pairwise_dataset(
            test_problem_alg,
            feature_cols=feature_cols,
            algorithms=algorithms,
            max_pairs_per_problem=None,
            add_reversed=ADD_REVERSED_PAIRS,
            random_state=RANDOM_SEED + fold,
        )

        if len(X_pair_train) == 0:
            raise RuntimeError("No pairwise training rows generated.")

        print(f"[{model_name}] fold {fold + 1}: pair train rows={len(X_pair_train)}, pair test rows={len(X_pair_test)}")

        model = safe_model_clone(model_template)
        model.fit(X_pair_train, y_pair_train)

        if len(X_pair_test):
            pair_eval = evaluate_pair_classifier_on_test(model, X_pair_test, y_pair_test)
            pair_eval["model"] = model_name
            pair_eval["fold"] = fold
            pair_eval["n_pair_train"] = len(X_pair_train)
            pair_eval["n_pair_test"] = len(X_pair_test)
            pair_metric_rows.append(pair_eval)

        selector_df, score_df = evaluate_selector_on_test(
            model=model,
            test_problem_alg_df=test_problem_alg,
            train_problem_alg_df=train_problem_alg,
            feature_cols=feature_cols,
            algorithms=algorithms,
            pair_cols=pair_cols,
            fold=fold,
            model_name=model_name,
        )
        selector_parts.append(selector_df)
        if not score_df.empty:
            score_parts.append(score_df)

    selector_df_all = pd.concat(selector_parts, ignore_index=True)
    score_df_all = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    pair_metrics_df = pd.DataFrame(pair_metric_rows)

    # Overall pair classification metrics.
    if not pair_metrics_df.empty:
        for m in ["pair_accuracy", "pair_roc_auc", "pair_log_loss"]:
            add_metric(metric_rows, model_name, "pair_classifier", m, float(pair_metrics_df[m].mean()))

    # Overall selector metrics.
    add_metric(metric_rows, model_name, "selector", "oracle_match_accuracy", float(selector_df_all["selected_is_oracle"].mean()))
    add_metric(metric_rows, model_name, "selector", "mean_normalized_regret_vs_oracle", float(selector_df_all["normalized_regret_vs_oracle"].mean()))
    add_metric(metric_rows, model_name, "selector", "median_normalized_regret_vs_oracle", float(selector_df_all["normalized_regret_vs_oracle"].median()))
    add_metric(metric_rows, model_name, "selector", "mean_fixed_baseline_normalized_regret_vs_oracle", float(selector_df_all["fixed_baseline_normalized_regret_vs_oracle"].mean()))
    add_metric(
        metric_rows,
        model_name,
        "selector",
        "mean_normalized_regret_improvement_over_fixed_baseline",
        float(selector_df_all["fixed_baseline_normalized_regret_vs_oracle"].mean() - selector_df_all["normalized_regret_vs_oracle"].mean()),
        note="positive means pairwise selector has lower regret than fixed baseline",
    )
    add_metric(metric_rows, model_name, "selector", "mean_relative_regret_vs_oracle", float(selector_df_all["relative_regret_vs_oracle"].mean()))
    add_metric(metric_rows, model_name, "selector", "median_relative_regret_vs_oracle", float(selector_df_all["relative_regret_vs_oracle"].median()))
    add_metric(metric_rows, model_name, "selector", "mean_ranking_spearman", float(selector_df_all["ranking_spearman"].mean()))
    add_metric(metric_rows, model_name, "selector", "mean_top3_overlap", float(selector_df_all["top3_overlap"].mean()))

    # Per-source metrics.
    for source, sub in selector_df_all.groupby("problem_type"):
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
        add_metric(metric_rows, model_name, "selector", "mean_ranking_spearman", float(sub["ranking_spearman"].mean()), source=source)
        add_metric(metric_rows, model_name, "selector", "mean_top3_overlap", float(sub["top3_overlap"].mean()), source=source)

    metrics_df = pd.DataFrame(metric_rows)
    elapsed = time.time() - start
    metrics_df["elapsed_seconds"] = elapsed

    print(f"[{model_name}] done in {elapsed:.1f}s.")
    return selector_df_all, score_df_all, pair_metrics_df, metrics_df


def get_metric(metrics_df, model, section, metric, source="ALL"):
    sub = metrics_df[
        (metrics_df["model"] == model)
        & (metrics_df["section"] == section)
        & (metrics_df["metric"] == metric)
        & (metrics_df["source"] == source)
    ]
    if len(sub) == 0:
        return np.nan
    return float(sub["value"].iloc[0])


def make_leaderboard(metrics_df):
    rows = []
    for model in sorted(metrics_df["model"].unique()):
        rows.append({
            "model": model,
            "pair_accuracy": get_metric(metrics_df, model, "pair_classifier", "pair_accuracy"),
            "pair_roc_auc": get_metric(metrics_df, model, "pair_classifier", "pair_roc_auc"),
            "pair_log_loss": get_metric(metrics_df, model, "pair_classifier", "pair_log_loss"),
            "oracle_match_accuracy": get_metric(metrics_df, model, "selector", "oracle_match_accuracy"),
            "mean_normalized_regret_vs_oracle": get_metric(metrics_df, model, "selector", "mean_normalized_regret_vs_oracle"),
            "mean_fixed_baseline_normalized_regret_vs_oracle": get_metric(metrics_df, model, "selector", "mean_fixed_baseline_normalized_regret_vs_oracle"),
            "mean_normalized_regret_improvement_over_fixed_baseline": get_metric(metrics_df, model, "selector", "mean_normalized_regret_improvement_over_fixed_baseline"),
            "mean_relative_regret_vs_oracle": get_metric(metrics_df, model, "selector", "mean_relative_regret_vs_oracle"),
            "mean_ranking_spearman": get_metric(metrics_df, model, "selector", "mean_ranking_spearman"),
            "mean_top3_overlap": get_metric(metrics_df, model, "selector", "mean_top3_overlap"),
        })

    lb = pd.DataFrame(rows)
    lb = lb.sort_values(
        ["mean_normalized_regret_vs_oracle", "oracle_match_accuracy", "mean_top3_overlap"],
        ascending=[True, False, False],
    )
    lb["rank_by_selector_regret"] = np.arange(1, len(lb) + 1)
    return lb


# ============================================================
# 8. Final fit and save
# ============================================================

def fit_final_pairwise_model(model_template, problem_alg_df, feature_cols, algorithms):
    X_pair, y_pair, pair_groups, meta = build_pairwise_dataset(
        problem_alg_df,
        feature_cols=feature_cols,
        algorithms=algorithms,
        max_pairs_per_problem=MAX_PAIRS_PER_PROBLEM,
        add_reversed=ADD_REVERSED_PAIRS,
        random_state=RANDOM_SEED,
    )
    model = safe_model_clone(model_template)
    model.fit(X_pair, y_pair)
    pair_cols = X_pair.columns.tolist()
    return model, pair_cols, len(X_pair)


def save_final_model_bundle(model_name, model, pair_feature_cols, feature_cols, algorithms, normalizer_params, leaderboard_row, n_pair_train):
    bundle = {
        "pairwise_classifier": model,
        "model_name": model_name,
        "model_type": "pairwise_algorithm_ranker",
        "pair_feature_cols": pair_feature_cols,
        "feature_cols": feature_cols,
        "algorithms": algorithms,
        "target_transform": "mixed_raw_log_per_problem_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_level": "problem_instance",
        "problem_key_definition": problem_key_cols(),
        "problem_normalizer_params": normalizer_params,
        "pairwise_training": {
            "add_reversed_pairs": ADD_REVERSED_PAIRS,
            "max_pairs_per_problem": MAX_PAIRS_PER_PROBLEM,
            "n_pair_train": n_pair_train,
        },
        "selection_note": (
            "For each problem, compare every algorithm pair. "
            "Each algorithm receives soft votes from predicted pairwise win probabilities. "
            "Select the algorithm with the highest total soft vote."
        ),
        "leaderboard_row": leaderboard_row,
    }

    path = os.path.join(MODEL_DIR, f"{model_name}_pairwise_ranker.joblib")
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
        os.path.join(PLOT_DIR, "pairwise_model_selector_regret.png"),
        "Pairwise model comparison: selector regret vs oracle",
        "Mean normalized regret vs oracle",
        higher_is_better=False,
    )

    barplot_metric(
        leaderboard,
        "oracle_match_accuracy",
        os.path.join(PLOT_DIR, "pairwise_model_oracle_match_accuracy.png"),
        "Pairwise model comparison: oracle match accuracy",
        "Oracle match accuracy",
        higher_is_better=True,
    )

    barplot_metric(
        leaderboard,
        "mean_ranking_spearman",
        os.path.join(PLOT_DIR, "pairwise_model_ranking_spearman.png"),
        "Pairwise model comparison: ranking Spearman",
        "Mean ranking Spearman",
        higher_is_better=True,
    )

    barplot_metric(
        leaderboard,
        "mean_top3_overlap",
        os.path.join(PLOT_DIR, "pairwise_model_top3_overlap.png"),
        "Pairwise model comparison: Top-3 overlap",
        "Mean Top-3 overlap",
        higher_is_better=True,
    )

    barplot_metric(
        leaderboard,
        "mean_normalized_regret_improvement_over_fixed_baseline",
        os.path.join(PLOT_DIR, "pairwise_model_improvement_over_fixed_baseline.png"),
        "Pairwise model comparison: improvement over fixed baseline",
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

    problem_alg_df = build_problem_alg_table(ela_df, perf_df, feature_cols)
    algorithms = sorted(problem_alg_df["algname"].astype(str).unique().tolist())

    print(f"Problem-alg rows: {len(problem_alg_df)}")
    print(f"Problems: {problem_alg_df['problem_key'].nunique()}")
    print(f"Features: {len(feature_cols)}")
    print(f"Algorithms: {len(algorithms)}")
    print("Rows by source:")
    print(problem_alg_df["problem_type"].value_counts())

    problem_alg_df.to_csv(os.path.join(OUT_DIR, "problem_algorithm_table.csv"), index=False)

    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(OUT_DIR, "algorithms.json"), "w") as f:
        json.dump(algorithms, f, indent=2)
    with open(os.path.join(OUT_DIR, "per_problem_normalizer_params.json"), "w") as f:
        json.dump(normalizer_params, f, indent=2)

    models = build_pairwise_models()
    print("Pairwise models to run:", list(models.keys()))

    all_selector_parts = []
    all_score_parts = []
    all_pair_metric_parts = []
    all_metric_parts = []

    for model_name, model_template in models.items():
        selector_df, score_df, pair_metrics_df, metrics_df = evaluate_model_cv(
            model_name=model_name,
            model_template=model_template,
            problem_alg_df=problem_alg_df,
            feature_cols=feature_cols,
            algorithms=algorithms,
        )

        all_selector_parts.append(selector_df)
        if not score_df.empty:
            all_score_parts.append(score_df)
        if not pair_metrics_df.empty:
            all_pair_metric_parts.append(pair_metrics_df)
        all_metric_parts.append(metrics_df)

        selector_df.to_csv(os.path.join(OUT_DIR, f"validation_selector_{model_name}.csv"), index=False)
        if not score_df.empty:
            score_df.to_csv(os.path.join(OUT_DIR, f"validation_pairwise_scores_{model_name}.csv"), index=False)
        if not pair_metrics_df.empty:
            pair_metrics_df.to_csv(os.path.join(OUT_DIR, f"validation_pair_classifier_metrics_{model_name}.csv"), index=False)
        metrics_df.to_csv(os.path.join(OUT_DIR, f"validation_metrics_{model_name}.csv"), index=False)

    all_selector = pd.concat(all_selector_parts, ignore_index=True)
    all_selector.to_csv(os.path.join(OUT_DIR, "validation_selector_all_pairwise_models.csv"), index=False)

    if all_score_parts:
        all_scores = pd.concat(all_score_parts, ignore_index=True)
        all_scores.to_csv(os.path.join(OUT_DIR, "validation_pairwise_scores_all_models.csv"), index=False)

    if all_pair_metric_parts:
        all_pair_metrics = pd.concat(all_pair_metric_parts, ignore_index=True)
        all_pair_metrics.to_csv(os.path.join(OUT_DIR, "validation_pair_classifier_metrics_all_models.csv"), index=False)

    all_metrics = pd.concat(all_metric_parts, ignore_index=True)
    all_metrics.to_csv(os.path.join(OUT_DIR, "validation_metrics_all_pairwise_models_long.csv"), index=False)

    leaderboard = make_leaderboard(all_metrics)
    leaderboard_path = os.path.join(OUT_DIR, "pairwise_model_comparison_leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"Saved: {leaderboard_path}")

    print("\n=== Pairwise leaderboard ===")
    print(leaderboard)

    make_plots(leaderboard)

    if SAVE_FINAL_MODELS:
        print("\n=== Fitting final pairwise models on all data ===")
        saved_paths = {}

        for model_name, model_template in models.items():
            model, pair_cols, n_pair_train = fit_final_pairwise_model(
                model_template=model_template,
                problem_alg_df=problem_alg_df,
                feature_cols=feature_cols,
                algorithms=algorithms,
            )
            row = leaderboard[leaderboard["model"] == model_name].iloc[0].to_dict()
            saved_paths[model_name] = save_final_model_bundle(
                model_name=model_name,
                model=model,
                pair_feature_cols=pair_cols,
                feature_cols=feature_cols,
                algorithms=algorithms,
                normalizer_params=normalizer_params,
                leaderboard_row=row,
                n_pair_train=n_pair_train,
            )

        if SAVE_BEST_MODEL_ALIAS:
            best_model_name = leaderboard.iloc[0]["model"]
            best_path = saved_paths[best_model_name]
            best_bundle = joblib.load(best_path)
            alias_path = os.path.join(MODEL_DIR, "best_pairwise_ranker.joblib")
            joblib.dump(best_bundle, alias_path)
            print(f"Saved best-model alias: {alias_path}")
            print(f"Best pairwise model by selector regret: {best_model_name}")

    config = {
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "target_transform": "mixed_raw_log_per_problem_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "add_reversed_pairs": ADD_REVERSED_PAIRS,
        "max_pairs_per_problem": MAX_PAIRS_PER_PROBLEM,
        "models_run": list(models.keys()),
        "primary_leaderboard_sort": "mean_normalized_regret_vs_oracle ascending, oracle_match_accuracy descending, mean_top3_overlap descending",
    }
    with open(os.path.join(OUT_DIR, "pairwise_comparison_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
