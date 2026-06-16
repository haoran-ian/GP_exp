# fmt: off
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
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

MODEL_SAVE_PATH = "data/Combined/models/bbob_mabbob_llm_mixed_auc_source_normalized_regressor_as_model.joblib"
OUT_DIR = "data/Combined/validation_mixed_auc_source_normalized_regressor"
PLOT_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 2. Config
# ============================================================

RANDOM_SEED = 42
N_SPLITS = 5
N_ESTIMATORS = 500
SAVE_FINAL_MODEL = True

# -----------------------------
# AUC handling
# -----------------------------
# Mixed target transform:
#   BBOB/MABBOB: use raw AUC directly;
#   LLM: filter abnormal raw AUC, then use log1p(raw AUC).
#
# After this source-specific transform, all sources are normalized separately.
AUC_MIN_POSITIVE = 1e-300

# LLM abnormal AUC filtering only.
LLM_AUC_ABS_MAX = 1e100
LLM_AUC_UPPER_QUANTILE = 0.995
DROP_LLM_ABNORMAL_AUC = True

# For BBOB/MABBOB, no abnormal finite AUC filtering is applied.
# Non-finite AUC rows are always removed because sklearn cannot fit NaN/inf.
DROP_NONFINITE_AUC_ONLY_FOR_BBOB_MABBOB = True

# -----------------------------
# Source-wise target normalization
# -----------------------------
# "minmax": target = (auc - source_min) / (source_max - source_min)
# "zscore": target = (auc - source_mean) / source_std
# "robust": target = (auc - source_median) / source_IQR
NORMALIZATION_METHOD = "minmax"
NORMALIZATION_EPS = 1e-12


META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim", "seed", "n_samples",
    "instance_id", "mabbob_instance_id", "llm_problem_id", "selection_method",
    "lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
    "source_dataset",
]


# ============================================================
# 3. Shared cleaning / key alignment
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


def drop_invalid_problem_rows(df, problem_type, stage):
    df = df.copy()
    n0 = len(df)

    if "FAILED" in df.columns:
        failed_mask = pd.to_numeric(df["FAILED"], errors="coerce").fillna(0) != 0
        df = df.loc[~failed_mask].copy()

    if "dim" not in df.columns:
        raise ValueError(f"{problem_type} {stage} table has no 'dim' column.")

    dim_num = pd.to_numeric(df["dim"], errors="coerce")
    valid_dim = np.isfinite(dim_num) & (dim_num > 0)
    df = df.loc[valid_dim].copy()
    df["dim"] = dim_num.loc[df.index].astype(int)

    n_drop = n0 - len(df)
    if n_drop > 0:
        print(f"[Clean] Dropped {n_drop} invalid {problem_type} {stage} rows; kept {len(df)} / {n0}.")
    return df


def ensure_problem_keys(df, problem_type):
    df = harmonize_feature_names(df.copy())
    df = drop_invalid_problem_rows(df, problem_type, stage="ELA")
    df["problem_type"] = problem_type

    if problem_type == "BBOB":
        df = df[df["fid"].between(1, 24)].copy()
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype(int)
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
        id_num = pd.to_numeric(df["mabbob_instance_id"], errors="coerce")
        df = df.loc[np.isfinite(id_num)].copy()
        df["mabbob_instance_id"] = id_num.loc[df.index].astype(int)
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
        id_num = pd.to_numeric(df["llm_problem_id"], errors="coerce")
        n0 = len(df)
        df = df.loc[np.isfinite(id_num)].copy()
        df["llm_problem_id"] = id_num.loc[df.index].astype(int)
        if len(df) < n0:
            print(f"[Clean] Dropped {n0 - len(df)} LLM ELA rows with invalid llm_problem_id.")
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


def clean_and_transform_value_by_source(df, problem_type):
    """
    Source-specific AUC handling before source-wise normalization.

    BBOB/MABBOB:
      - no log transform;
      - no abnormal finite AUC filtering;
      - only remove non-finite auc_mean rows.

    LLM:
      - remove non-finite/non-positive AUC;
      - remove abnormal large AUC values using a global cap and/or source quantile;
      - convert raw AUC to transformed_auc = log1p(raw AUC).

    The downstream source-wise normalizer is fitted on transformed_auc, not
    directly on auc_mean.
    """
    df = df.copy()
    n0 = len(df)

    df["auc_mean"] = pd.to_numeric(df["auc_mean"], errors="coerce")
    df["auc_filter_reason"] = "kept"

    if problem_type in ["BBOB", "MABBOB"]:
        invalid = ~np.isfinite(df["auc_mean"])
        df.loc[invalid, "auc_filter_reason"] = "non_finite_auc"

        invalid_rows = df.loc[invalid].copy()
        if len(invalid_rows) > 0:
            invalid_path = os.path.join(OUT_DIR, f"auc_nonfinite_rows_{problem_type}.csv")
            invalid_rows.to_csv(invalid_path, index=False)

        df = df.loc[~invalid].copy()
        df["auc_transform"] = "raw"
        df["transformed_auc"] = df["auc_mean"].astype(float)

        report = pd.DataFrame([
            {"problem_type": problem_type, "auc_filter_reason": "kept_finite_raw_auc", "n_rows": len(df)},
            {"problem_type": problem_type, "auc_filter_reason": "dropped_nonfinite_auc", "n_rows": len(invalid_rows)},
        ])
        report_path = os.path.join(OUT_DIR, f"auc_mixed_filter_report_{problem_type}.csv")
        report.to_csv(report_path, index=False)

        if len(invalid_rows) > 0:
            print(f"[AUC clean] {problem_type}: dropped {len(invalid_rows)} non-finite AUC rows; kept {len(df)} / {n0}.")
        else:
            print(f"[AUC clean] {problem_type}: no abnormal filtering; kept {len(df)} / {n0} finite AUC rows.")

        return df

    if problem_type == "LLM":
        invalid = (~np.isfinite(df["auc_mean"])) | (df["auc_mean"] <= AUC_MIN_POSITIVE)
        df.loc[invalid, "auc_filter_reason"] = "non_finite_or_non_positive"

        if LLM_AUC_ABS_MAX is not None:
            too_large_global = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > LLM_AUC_ABS_MAX)
            df.loc[too_large_global, "auc_filter_reason"] = f"above_llm_global_cap_{LLM_AUC_ABS_MAX:.1e}"

        if LLM_AUC_UPPER_QUANTILE is not None:
            valid_for_q = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > AUC_MIN_POSITIVE)
            if valid_for_q.any():
                q = df.loc[valid_for_q, "auc_mean"].quantile(LLM_AUC_UPPER_QUANTILE)
                if np.isfinite(q):
                    too_large_q = valid_for_q & (df["auc_mean"] > q)
                    df.loc[too_large_q, "auc_filter_reason"] = f"above_llm_q{LLM_AUC_UPPER_QUANTILE}"

        abnormal = df["auc_filter_reason"] != "kept"

        report = (
            df.groupby(["problem_type", "auc_filter_reason"])
            .size()
            .reset_index(name="n_rows")
            .sort_values(["problem_type", "auc_filter_reason"])
        )
        report_path = os.path.join(OUT_DIR, f"auc_mixed_filter_report_{problem_type}.csv")
        report.to_csv(report_path, index=False)

        abnormal_path = os.path.join(OUT_DIR, f"auc_abnormal_rows_{problem_type}.csv")
        df.loc[abnormal].to_csv(abnormal_path, index=False)

        if DROP_LLM_ABNORMAL_AUC:
            df = df.loc[~abnormal].copy()
            print(f"[AUC clean] {problem_type}: dropped {n0 - len(df)} abnormal AUC rows; kept {len(df)} / {n0}.")
        else:
            # Keep finite positive rows, but clip extreme values.
            df = df.loc[~invalid].copy()
            if LLM_AUC_ABS_MAX is not None:
                df["auc_mean"] = df["auc_mean"].clip(upper=LLM_AUC_ABS_MAX)
            if LLM_AUC_UPPER_QUANTILE is not None:
                q = df["auc_mean"].quantile(LLM_AUC_UPPER_QUANTILE)
                if np.isfinite(q):
                    df["auc_mean"] = df["auc_mean"].clip(upper=q)
            print(f"[AUC clean] {problem_type}: clipped abnormal AUC rows; kept {len(df)} / {n0}.")

        df["auc_transform"] = "log1p"
        df["transformed_auc"] = np.log1p(np.maximum(df["auc_mean"].to_numpy(dtype=float), AUC_MIN_POSITIVE))
        return df

    raise ValueError(problem_type)


def ensure_perf_keys(df, problem_type):
    df = drop_invalid_problem_rows(df.copy(), problem_type, stage="performance")
    df["problem_type"] = problem_type

    if problem_type == "BBOB":
        df = df[df["fid"].between(1, 24)].copy()
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype(int)
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
        id_num = pd.to_numeric(df["mabbob_instance_id"], errors="coerce")
        df = df.loc[np.isfinite(id_num)].copy()
        df["mabbob_instance_id"] = id_num.loc[df.index].astype(int)
        df["fid"] = -100
        df["iid"] = df["mabbob_instance_id"].astype(int)
        df["problem_name"] = df["iid"].apply(lambda x: f"MABBOB_{int(x)}")

    elif problem_type == "LLM":
        if "llm_problem_id" not in df.columns:
            if "iid" in df.columns:
                df["llm_problem_id"] = df["iid"]
            else:
                raise ValueError("LLM performance must contain llm_problem_id or iid.")
        id_num = pd.to_numeric(df["llm_problem_id"], errors="coerce")
        n0 = len(df)
        df = df.loc[np.isfinite(id_num)].copy()
        df["llm_problem_id"] = id_num.loc[df.index].astype(int)
        if len(df) < n0:
            print(f"[Clean] Dropped {n0 - len(df)} LLM performance rows with invalid llm_problem_id.")
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
    df = clean_and_transform_value_by_source(df, problem_type)
    df["auc_mean"] = pd.to_numeric(df["auc_mean"], errors="coerce").astype(float)
    df["source_dataset"] = problem_type
    return df


def get_feature_cols(ela_df):
    excluded = set(META_COLS)
    feature_cols = []
    for c in ela_df.columns:
        if c in excluded:
            continue
        if c.endswith(".FAILED"):
            continue
        if c.endswith(".ERROR") or c == "ERROR" or c == "FAILED":
            continue
        if pd.api.types.is_numeric_dtype(ela_df[c]):
            feature_cols.append(c)

    X = ela_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    all_empty = X.columns[X.isna().all()].tolist()
    feature_cols = [c for c in feature_cols if c not in all_empty]
    if feature_cols:
        nunique = X[feature_cols].nunique(dropna=True)
        const_cols = nunique[nunique <= 1].index.tolist()
        feature_cols = [c for c in feature_cols if c not in const_cols]
    return sorted(feature_cols)


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


def problem_key_cols():
    return ["problem_type", "fid", "iid", "dim"]


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
# 4. Source-wise normalization
# ============================================================

def fit_source_normalizers(perf_df):
    params = {}
    for source, sub in perf_df.groupby("problem_type"):
        values = sub["transformed_auc"].to_numpy(dtype=float)
        if NORMALIZATION_METHOD == "minmax":
            vmin, vmax = float(np.min(values)), float(np.max(values))
            scale = vmax - vmin
            if not np.isfinite(scale) or scale <= NORMALIZATION_EPS:
                scale = 1.0
            params[source] = {"method": "minmax", "min": vmin, "max": vmax, "scale": scale}
        elif NORMALIZATION_METHOD == "zscore":
            mean, std = float(np.mean(values)), float(np.std(values))
            if not np.isfinite(std) or std <= NORMALIZATION_EPS:
                std = 1.0
            params[source] = {"method": "zscore", "mean": mean, "std": std}
        elif NORMALIZATION_METHOD == "robust":
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            iqr = float(q75 - q25)
            if not np.isfinite(iqr) or iqr <= NORMALIZATION_EPS:
                iqr = 1.0
            params[source] = {"method": "robust", "median": median, "q25": float(q25), "q75": float(q75), "iqr": iqr}
        else:
            raise ValueError(f"Unknown NORMALIZATION_METHOD: {NORMALIZATION_METHOD}")
    return params


def transform_value_by_source(auc, source, normalizer_params):
    p = normalizer_params[source]
    auc = np.asarray(auc, dtype=float)
    if p["method"] == "minmax":
        return (auc - p["min"]) / p["scale"]
    if p["method"] == "zscore":
        return (auc - p["mean"]) / p["std"]
    if p["method"] == "robust":
        return (auc - p["median"]) / p["iqr"]
    raise ValueError(p["method"])


def inverse_transform_value_by_source(target, source, normalizer_params):
    p = normalizer_params[source]
    target = np.asarray(target, dtype=float)
    if p["method"] == "minmax":
        return target * p["scale"] + p["min"]
    if p["method"] == "zscore":
        return target * p["std"] + p["mean"]
    if p["method"] == "robust":
        return target * p["iqr"] + p["median"]
    raise ValueError(p["method"])


def add_source_normalized_target(perf_df, normalizer_params):
    perf_df = perf_df.copy()
    perf_df["target_auc"] = np.nan
    for source, idx in perf_df.groupby("problem_type").groups.items():
        perf_df.loc[idx, "target_auc"] = transform_value_by_source(
            perf_df.loc[idx, "transformed_auc"].to_numpy(dtype=float), source, normalizer_params
        )
    return perf_df


# ============================================================
# 5. Regressor training table
# ============================================================

def build_regressor_train_table(ela_df, perf_df):
    return pd.merge(ela_df, perf_df, on=problem_key_cols(), how="inner", suffixes=("", "_perf"))


def make_regressor_X(train_df_reg, feature_cols, all_algorithms=None):
    X_base = clean_X(train_df_reg[feature_cols])
    alg_dummies = pd.get_dummies(train_df_reg["algname"].astype(str), prefix="algname")
    X = pd.concat([X_base.reset_index(drop=True), alg_dummies.reset_index(drop=True)], axis=1)
    if all_algorithms is not None:
        all_alg_cols = [f"algname_{a}" for a in all_algorithms]
        for c in all_alg_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols + all_alg_cols]
    return X


# ============================================================
# 6. Metrics
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


def add_metric(rows, section, metric, value, source="ALL", note=""):
    rows.append({"section": section, "source": source, "metric": metric, "value": value, "note": note})


def train_fold_algorithm_mean_baseline(train_rows, test_rows):
    global_mean = train_rows["target_auc"].mean()
    alg_mean = train_rows.groupby("algname")["target_auc"].mean().to_dict()
    return test_rows["algname"].map(alg_mean).fillna(global_mean).to_numpy(dtype=float)


def selector_validation_from_regressor_fold(test_rows, pred_target, train_rows, fold):
    tmp = test_rows.copy()
    tmp["pred_target_auc"] = pred_target
    rows = []
    train_alg_mean = train_rows.groupby("algname")["target_auc"].mean()
    fixed_alg = train_alg_mean.idxmin()

    for key, g in tmp.groupby(problem_key_cols()):
        g = g.copy()
        pred_row = g.loc[g["pred_target_auc"].idxmin()]
        oracle_row = g.loc[g["auc_mean"].idxmin()]
        fixed_rows = g[g["algname"] == fixed_alg]
        if len(fixed_rows) > 0:
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
            "fold": int(fold),
            "problem_type": key[0],
            "fid": int(key[1]),
            "iid": int(key[2]),
            "dim": int(key[3]),
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
            "fixed_baseline_relative_regret_vs_oracle": fixed_actual_auc / oracle_actual_auc - 1.0 if oracle_actual_auc > AUC_MIN_POSITIVE and np.isfinite(fixed_actual_auc) else np.nan,
            "selected_is_oracle": pred_row["algname"] == oracle_row["algname"],
            "n_algorithms_available": int(len(g)),
        })
    return pd.DataFrame(rows)


# ============================================================
# 7. Cross-validation
# ============================================================

def validate_regressor(train_df_reg, feature_cols, normalizer_params):
    print("\n=== Mixed-transformed source-normalized AUC regressor validation ===")
    all_algorithms = sorted(train_df_reg["algname"].astype(str).unique().tolist())
    y = train_df_reg["target_auc"].astype(float).to_numpy()
    y_raw = train_df_reg["auc_mean"].astype(float).to_numpy()
    X = make_regressor_X(train_df_reg, feature_cols, all_algorithms=all_algorithms)
    groups = make_groups(train_df_reg)
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for regressor GroupKFold validation.")

    reg_template = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features="log2", n_jobs=-1, random_state=RANDOM_SEED)
    cv = GroupKFold(n_splits=n_splits)
    pred_target = np.full(len(train_df_reg), np.nan)
    baseline_target = np.full(len(train_df_reg), np.nan)
    fold_id = np.full(len(train_df_reg), -1, dtype=int)
    selector_parts = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"[Regressor CV] fold {fold + 1}/{n_splits}: train={len(train_idx)}, test={len(test_idx)}")
        reg = clone(reg_template)
        reg.fit(X.iloc[train_idx], y[train_idx])
        fold_pred = reg.predict(X.iloc[test_idx])
        pred_target[test_idx] = fold_pred
        train_rows = train_df_reg.iloc[train_idx].copy()
        test_rows = train_df_reg.iloc[test_idx].copy()
        baseline_target[test_idx] = train_fold_algorithm_mean_baseline(train_rows, test_rows)
        fold_id[test_idx] = fold
        selector_parts.append(selector_validation_from_regressor_fold(test_rows, fold_pred, train_rows, fold))

    pred_df = train_df_reg[problem_key_cols() + ["problem_name", "algname", "auc_mean", "target_auc"]].copy()
    pred_df["fold"] = fold_id
    pred_df["pred_target_auc"] = pred_target
    pred_df["baseline_alg_mean_pred_target_auc"] = baseline_target
    pred_df["pred_transformed_auc_backtransformed"] = np.nan
    pred_df["baseline_transformed_auc_backtransformed"] = np.nan
    for source, idx in pred_df.groupby("problem_type").groups.items():
        pred_df.loc[idx, "pred_transformed_auc_backtransformed"] = inverse_transform_value_by_source(
            pred_df.loc[idx, "pred_target_auc"].to_numpy(float),
            source,
            normalizer_params,
        )
        pred_df.loc[idx, "baseline_transformed_auc_backtransformed"] = inverse_transform_value_by_source(
            pred_df.loc[idx, "baseline_alg_mean_pred_target_auc"].to_numpy(float),
            source,
            normalizer_params,
        )

    # Convert the source-specific transformed scale back to raw AUC for diagnostics.
    # BBOB/MABBOB transformed_auc is raw AUC; LLM transformed_auc is log1p(raw AUC).
    pred_df["pred_raw_auc_backtransformed"] = np.nan
    pred_df["baseline_raw_auc_backtransformed"] = np.nan

    for source, idx in pred_df.groupby("problem_type").groups.items():
        pred_t = pred_df.loc[idx, "pred_transformed_auc_backtransformed"].to_numpy(dtype=float)
        base_t = pred_df.loc[idx, "baseline_transformed_auc_backtransformed"].to_numpy(dtype=float)

        if source == "LLM":
            pred_df.loc[idx, "pred_raw_auc_backtransformed"] = np.expm1(np.clip(pred_t, -745, 700))
            pred_df.loc[idx, "baseline_raw_auc_backtransformed"] = np.expm1(np.clip(base_t, -745, 700))
        else:
            pred_df.loc[idx, "pred_raw_auc_backtransformed"] = pred_t
            pred_df.loc[idx, "baseline_raw_auc_backtransformed"] = base_t

    pred_df["abs_normalized_error"] = np.abs(pred_df["target_auc"] - pred_df["pred_target_auc"])
    pred_df["baseline_abs_normalized_error"] = np.abs(pred_df["target_auc"] - pred_df["baseline_alg_mean_pred_target_auc"])
    pred_df["abs_raw_error"] = np.abs(pred_df["auc_mean"] - pred_df["pred_raw_auc_backtransformed"])
    pred_df["baseline_abs_raw_error"] = np.abs(pred_df["auc_mean"] - pred_df["baseline_raw_auc_backtransformed"])
    pred_df.to_csv(os.path.join(OUT_DIR, "validation_regressor_predictions.csv"), index=False)

    selector_df = pd.concat(selector_parts, ignore_index=True)
    selector_df.to_csv(os.path.join(OUT_DIR, "validation_selector_from_regressor.csv"), index=False)

    metric_rows = []
    add_metric(metric_rows, "regressor_normalized_target", "r2", safe_r2(y, pred_target))
    add_metric(metric_rows, "regressor_normalized_target", "mae", float(mean_absolute_error(y, pred_target)))
    add_metric(metric_rows, "regressor_normalized_target", "rmse", rmse(y, pred_target))
    add_metric(metric_rows, "regressor_normalized_target", "spearman", safe_spearman(y, pred_target))
    add_metric(metric_rows, "regressor_normalized_target", "baseline_alg_mean_mae", float(mean_absolute_error(y, baseline_target)))
    add_metric(metric_rows, "regressor_normalized_target", "baseline_alg_mean_rmse", rmse(y, baseline_target))
    add_metric(metric_rows, "regressor_normalized_target", "mae_improvement_over_alg_mean_baseline", float(mean_absolute_error(y, baseline_target) - mean_absolute_error(y, pred_target)), note="positive means lower normalized-target MAE than algorithm-mean baseline")
    add_metric(metric_rows, "regressor_raw_diagnostic", "mae", float(mean_absolute_error(y_raw, pred_df["pred_raw_auc_backtransformed"])))
    add_metric(metric_rows, "regressor_raw_diagnostic", "baseline_alg_mean_mae", float(mean_absolute_error(y_raw, pred_df["baseline_raw_auc_backtransformed"])))
    add_metric(metric_rows, "selector_from_regressor", "oracle_match_accuracy", float(selector_df["selected_is_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "mean_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "median_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].median()))
    add_metric(metric_rows, "selector_from_regressor", "mean_fixed_baseline_normalized_regret_vs_oracle", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "mean_normalized_regret_improvement_over_fixed_baseline", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean() - selector_df["normalized_regret_vs_oracle"].mean()), note="positive means model selector has lower normalized-regret than fixed baseline")
    add_metric(metric_rows, "selector_from_regressor", "mean_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "median_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].median()))

    for source, sub in pred_df.groupby("problem_type"):
        yt = sub["target_auc"].to_numpy(float)
        yp = sub["pred_target_auc"].to_numpy(float)
        yb = sub["baseline_alg_mean_pred_target_auc"].to_numpy(float)
        add_metric(metric_rows, "regressor_normalized_target", "r2", safe_r2(yt, yp), source=source)
        add_metric(metric_rows, "regressor_normalized_target", "mae", float(mean_absolute_error(yt, yp)), source=source)
        add_metric(metric_rows, "regressor_normalized_target", "rmse", rmse(yt, yp), source=source)
        add_metric(metric_rows, "regressor_normalized_target", "spearman", safe_spearman(yt, yp), source=source)
        add_metric(metric_rows, "regressor_normalized_target", "baseline_alg_mean_mae", float(mean_absolute_error(yt, yb)), source=source)
        add_metric(metric_rows, "regressor_normalized_target", "mae_improvement_over_alg_mean_baseline", float(mean_absolute_error(yt, yb) - mean_absolute_error(yt, yp)), source=source, note="positive means lower normalized-target MAE than algorithm-mean baseline")

    for source, sub in selector_df.groupby("problem_type"):
        add_metric(metric_rows, "selector_from_regressor", "oracle_match_accuracy", float(sub["selected_is_oracle"].mean()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "mean_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].mean()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "median_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].median()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "mean_fixed_baseline_normalized_regret_vs_oracle", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "mean_normalized_regret_improvement_over_fixed_baseline", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean() - sub["normalized_regret_vs_oracle"].mean()), source=source, note="positive means model selector has lower normalized-regret than fixed baseline")
        add_metric(metric_rows, "selector_from_regressor", "mean_relative_regret_vs_oracle", float(sub["relative_regret_vs_oracle"].mean()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "median_relative_regret_vs_oracle", float(sub["relative_regret_vs_oracle"].median()), source=source)

    return reg_template, all_algorithms, X.columns.tolist(), pred_df, selector_df, pd.DataFrame(metric_rows)


# ============================================================
# 8. Plots
# ============================================================

def plot_normalized_pred_vs_true(reg_pred_df):
    fig, ax = plt.subplots(figsize=(7, 6))
    source_order = ["BBOB", "MABBOB", "LLM"]
    colors = {"BBOB": "#1f77b4", "MABBOB": "#ff7f0e", "LLM": "#2ca02c"}
    plotted = []
    for source in source_order:
        sub = reg_pred_df[reg_pred_df["problem_type"] == source]
        if sub.empty:
            continue
        ax.scatter(sub["target_auc"], sub["pred_target_auc"], s=12, alpha=0.42, color=colors.get(source), label=f"{source} (n={len(sub)})")
        plotted.append(source)
    for source in sorted(set(reg_pred_df["problem_type"].unique()) - set(plotted)):
        sub = reg_pred_df[reg_pred_df["problem_type"] == source]
        ax.scatter(sub["target_auc"], sub["pred_target_auc"], s=12, alpha=0.42, label=f"{source} (n={len(sub)})")
    x = reg_pred_df["target_auc"].to_numpy(float)
    y = reg_pred_df["pred_target_auc"].to_numpy(float)
    lo, hi = min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", label="Ideal prediction")
    ax.set_xlabel("True mixed-transformed source-normalized AUC")
    ax.set_ylabel("Predicted mixed-transformed source-normalized AUC")
    ax.set_title(f"Cross-validated regressor predictions ({NORMALIZATION_METHOD} normalized target)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "regressor_source_normalized_target_pred_vs_true.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_normalized_abs_error_by_source(reg_pred_df):
    sources = sorted(reg_pred_df["problem_type"].unique())
    data = [reg_pred_df.loc[reg_pred_df["problem_type"] == s, "abs_normalized_error"].dropna().values for s in sources]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=sources, showfliers=False)
    ax.set_ylabel("Absolute mixed-transformed source-normalized AUC prediction error")
    ax.set_title("Regressor normalized-target absolute error by source")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "regressor_source_normalized_abs_error_by_source.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_selector_normalized_regret_by_source(selector_df):
    sources = sorted(selector_df["problem_type"].unique())
    data_model = [selector_df.loc[selector_df["problem_type"] == s, "normalized_regret_vs_oracle"].dropna().values for s in sources]
    data_base = [selector_df.loc[selector_df["problem_type"] == s, "fixed_baseline_normalized_regret_vs_oracle"].dropna().values for s in sources]
    fig, ax = plt.subplots(figsize=(9, 5))
    positions_model = np.arange(len(sources)) * 2.0 + 1.0
    positions_base = positions_model + 0.7
    bp_model = ax.boxplot(data_model, positions=positions_model, widths=0.55, showfliers=False, patch_artist=True)
    bp_base = ax.boxplot(data_base, positions=positions_base, widths=0.55, showfliers=False, patch_artist=True)
    for b in bp_model["boxes"]:
        b.set(facecolor="#1f77b4", alpha=0.55)
    for b in bp_base["boxes"]:
        b.set(facecolor="#ff7f0e", alpha=0.55)
    ax.plot([], [], color="#1f77b4", linewidth=8, alpha=0.55, label="Regressor selector")
    ax.plot([], [], color="#ff7f0e", linewidth=8, alpha=0.55, label="Fixed baseline")
    ax.set_xticks(positions_model + 0.35)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Source-normalized regret vs oracle")
    ax.set_title("Algorithm-selection normalized regret")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "selector_source_normalized_regret_by_source.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_selector_normalized_auc_by_source(selector_df):
    sources = sorted(selector_df["problem_type"].unique())
    data_oracle = [selector_df.loc[selector_df["problem_type"] == s, "oracle_actual_target_auc"].dropna().values for s in sources]
    data_model = [selector_df.loc[selector_df["problem_type"] == s, "pred_selected_actual_target_auc"].dropna().values for s in sources]
    data_base = [selector_df.loc[selector_df["problem_type"] == s, "fixed_baseline_actual_target_auc"].dropna().values for s in sources]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions_oracle = np.arange(len(sources)) * 3.0 + 1.0
    positions_model = positions_oracle + 0.7
    positions_base = positions_oracle + 1.4
    colors = {"Oracle": "#2ca02c", "Regressor selector": "#1f77b4", "Fixed baseline": "#ff7f0e"}
    def colored_boxplot(data, positions, color, label):
        ax.boxplot(data, positions=positions, widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=color, edgecolor="black", alpha=0.55),
                   medianprops=dict(color="black", linewidth=1.4),
                   whiskerprops=dict(color="black", linewidth=1.0),
                   capprops=dict(color="black", linewidth=1.0))
        ax.plot([], [], color=color, linewidth=8, alpha=0.55, label=label)
    colored_boxplot(data_oracle, positions_oracle, colors["Oracle"], "Best Algorithm per Problem")
    colored_boxplot(data_model, positions_model, colors["Regressor selector"], "Selected Best Algorithm per Problem")
    colored_boxplot(data_base, positions_base, colors["Fixed baseline"], "Best Algorithm Overall")
    ax.set_xticks(positions_oracle + 0.7)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Actual mixed-transformed source-normalized AUC of selected algorithm")
    ax.set_title("Actual normalized AUC: oracle vs regressor selector vs fixed baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "selector_actual_source_normalized_auc_by_source.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_auc_distribution_after_filtering(train_df_reg):
    fig, ax = plt.subplots(figsize=(8, 5))
    for source, sub in train_df_reg.groupby("problem_type"):
        vals = sub["target_auc"].to_numpy(float)
        ax.hist(vals, bins=50, alpha=0.45, label=f"{source} (n={len(sub)})")
    ax.set_xlabel("Mixed-transformed source-normalized AUC")
    ax.set_ylabel("Count")
    ax.set_title("AUC target distribution after source-wise normalization")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "auc_distribution_source_normalized_no_filter.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def make_plots(reg_pred_df, selector_df, train_df_reg):
    plot_auc_distribution_after_filtering(train_df_reg)
    plot_normalized_pred_vs_true(reg_pred_df)
    plot_normalized_abs_error_by_source(reg_pred_df)
    plot_selector_normalized_regret_by_source(selector_df)
    plot_selector_normalized_auc_by_source(selector_df)


# ============================================================
# 9. Final fit and save
# ============================================================

def final_fit_and_save(train_df_reg, feature_cols, reg_template, all_algorithms, reg_feature_cols, validation_meta, normalizer_params):
    print("\n=== Final fit on all mixed-transformed source-normalized data with LLM log transform and LLM abnormal-AUC filtering ===")
    X_reg = make_regressor_X(train_df_reg, feature_cols, all_algorithms=all_algorithms)
    X_reg = X_reg[reg_feature_cols]
    y_reg = train_df_reg["target_auc"].astype(float).values
    reg = clone(reg_template)
    reg.fit(X_reg, y_reg)
    bundle = {
        "regressor": reg,
        "feature_cols": feature_cols,
        "reg_feature_cols": reg_feature_cols,
        "algorithms": all_algorithms,
        "target_transform": "mixed_raw_log_source_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "normalizer_params": normalizer_params,
        "auc_min_positive": AUC_MIN_POSITIVE,
        "inverse_transform_note": "Use inverse_transform_value_by_source(pred, source, normalizer_params) to convert normalized predictions back to the source-specific transformed scale. For LLM, apply expm1 to recover raw AUC; for BBOB/MABBOB the transformed scale is raw AUC. Algorithm selection can use predicted normalized AUC directly because all source-specific transforms are monotonic.",
        "meta": validation_meta,
    }
    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"Saved final mixed-transformed source-normalized AUC regressor AS model bundle to: {MODEL_SAVE_PATH}")


# ============================================================
# 10. Main
# ============================================================

def main():
    ela_df, perf_df = load_all_data()
    normalizer_params = fit_source_normalizers(perf_df)
    perf_df = add_source_normalized_target(perf_df, normalizer_params)
    feature_cols = get_feature_cols(ela_df)

    print("\n=== Data summary after mixed AUC transform and source-wise normalization ===")
    print(f"ELA rows: {len(ela_df)}")
    print(f"Performance rows: {len(perf_df)}")
    print(f"Feature cols: {len(feature_cols)}")
    print(f"Normalization method: {NORMALIZATION_METHOD}")
    print("\nELA source counts:")
    print(ela_df["problem_type"].value_counts())
    print("\nPerformance source counts with LLM log transform and LLM abnormal-AUC filtering:")
    print(perf_df["problem_type"].value_counts())

    if len(feature_cols) == 0:
        raise RuntimeError("No usable numeric ELA features after cleaning.")

    normalizer_path = os.path.join(OUT_DIR, "source_auc_normalizer_params.json")
    with open(normalizer_path, "w") as f:
        json.dump(normalizer_params, f, indent=2)
    print(f"Saved: {normalizer_path}")

    train_df_reg = build_regressor_train_table(ela_df, perf_df)
    print("\n=== Matched regressor training table ===")
    print(f"Regressor rows: {len(train_df_reg)}")
    print("\nRegressor rows by source:")
    print(train_df_reg["problem_type"].value_counts())

    train_df_reg[problem_key_cols() + ["problem_name", "algname", "auc_mean", "auc_transform", "transformed_auc", "target_auc"]].to_csv(os.path.join(OUT_DIR, "training_regressor_table_keys.csv"), index=False)

    target_summary = train_df_reg.groupby("problem_type").agg(
        n_rows=("auc_mean", "count"),
        auc_min=("auc_mean", "min"),
        auc_median=("auc_mean", "median"),
        auc_mean=("auc_mean", "mean"),
        auc_max=("auc_mean", "max"),
        transformed_min=("transformed_auc", "min"),
        transformed_median=("transformed_auc", "median"),
        transformed_mean=("transformed_auc", "mean"),
        transformed_max=("transformed_auc", "max"),
        target_min=("target_auc", "min"),
        target_median=("target_auc", "median"),
        target_mean=("target_auc", "mean"),
        target_max=("target_auc", "max"),
    ).reset_index()
    target_summary_path = os.path.join(OUT_DIR, "target_auc_summary_after_source_normalization_no_filter.csv")
    target_summary.to_csv(target_summary_path, index=False)
    print(f"Saved: {target_summary_path}")

    reg_template, all_algorithms, reg_feature_cols, reg_pred_df, selector_df, metrics = validate_regressor(train_df_reg, feature_cols, normalizer_params)
    metrics_path = os.path.join(OUT_DIR, "validation_metrics_summary.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    make_plots(reg_pred_df, selector_df, train_df_reg)

    validation_meta = {
        "model_type": "regressor_only_algorithm_selection",
        "target_transform": "mixed_raw_log_source_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "normalizer_params_file": normalizer_path,
        "auc_handling": {
            "BBOB": {"transform": "raw", "abnormal_auc_filtering": False, "drop_nonfinite_auc_only": True},
            "MABBOB": {"transform": "raw", "abnormal_auc_filtering": False, "drop_nonfinite_auc_only": True},
            "LLM": {
                "transform": "log1p",
                "abnormal_auc_filtering": True,
                "llm_auc_abs_max": LLM_AUC_ABS_MAX,
                "llm_auc_upper_quantile": LLM_AUC_UPPER_QUANTILE,
                "drop_llm_abnormal_auc": DROP_LLM_ABNORMAL_AUC,
            },
        },
        "train_sources": ["BBOB", "MABBOB", "LLM"],
        "regressor_rows": int(len(train_df_reg)),
        "n_features": int(len(feature_cols)),
        "n_algorithms": int(len(all_algorithms)),
        "n_splits": int(min(N_SPLITS, len(np.unique(make_groups(train_df_reg))))),
        "cv_group_definition": "BBOB grouped by fid; MABBOB/LLM grouped by iid",
        "validation_metrics_file": metrics_path,
        "validation_outputs_dir": OUT_DIR,
        "target_summary_file": target_summary_path,
    }
    for _, row in metrics.iterrows():
        if row["source"] == "ALL":
            key = f"{row['section']}.{row['metric']}"
            validation_meta[key] = None if pd.isna(row["value"]) else float(row["value"])

    meta_path = os.path.join(OUT_DIR, "validation_meta.json")
    with open(meta_path, "w") as f:
        json.dump(validation_meta, f, indent=2)
    print(f"Saved: {meta_path}")

    if SAVE_FINAL_MODEL:
        final_fit_and_save(train_df_reg, feature_cols, reg_template, all_algorithms, reg_feature_cols, validation_meta, normalizer_params)
    print("\nDone.")


if __name__ == "__main__":
    main()
