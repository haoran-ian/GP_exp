
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

MODEL_SAVE_PATH = "data/Combined/models/bbob_mabbob_llm_log_auc_regressor_as_model.joblib"
OUT_DIR = "data/Combined/validation_log_auc_regressor"
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
# Abnormal AUC handling
# -----------------------------
# Basic validity
AUC_MIN_POSITIVE = 1e-300

# Global hard cap. Values above this are usually numerical explosions.
# Set to None to disable.
AUC_ABS_MAX = 1e100

# Per-source quantile clipping/filtering.
# Example: 0.995 removes the largest 0.5% AUC rows inside each source.
# Set to None to disable.
AUC_SOURCE_UPPER_QUANTILE = 0.995

# Per-problem quantile filtering. This is often useful because a single
# problem × algorithm can explode while other algorithms on the same problem
# remain usable. Set to None to disable.
AUC_PROBLEM_UPPER_QUANTILE = None

# If True, rows with abnormal AUC are removed.
# If False, abnormal AUC is clipped to the threshold.
DROP_ABNORMAL_AUC = True

# Train target:
#   log1p: recommended. Fits log(1 + AUC).
#   raw: original behavior, not recommended for heavy-tailed AUC.
TARGET_TRANSFORM = "log1p"

# Selector validation metric:
#   log_regret = log1p(selected_auc) - log1p(oracle_auc)
#   relative_regret = selected_auc / oracle_auc - 1, only stable when oracle_auc > 0
USE_LOG_REGRET = True


META_COLS = [
    "problem_type",
    "problem_name",
    "fid",
    "iid",
    "dim",
    "seed",
    "n_samples",
    "instance_id",
    "mabbob_instance_id",
    "llm_problem_id",
    "selection_method",
    "lower_bound_min",
    "lower_bound_max",
    "upper_bound_min",
    "upper_bound_max",
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
    """
    Robust numeric cleaning for tree models.
    """
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


def transform_target_auc(auc):
    auc = np.asarray(auc, dtype=float)
    auc = np.maximum(auc, AUC_MIN_POSITIVE)

    if TARGET_TRANSFORM == "log1p":
        return np.log1p(auc)
    if TARGET_TRANSFORM == "raw":
        return auc

    raise ValueError(f"Unknown TARGET_TRANSFORM: {TARGET_TRANSFORM}")


def inverse_transform_target(y):
    y = np.asarray(y, dtype=float)

    if TARGET_TRANSFORM == "log1p":
        # Guard against exp overflow in diagnostics.
        y = np.clip(y, -745, 700)
        return np.expm1(y)
    if TARGET_TRANSFORM == "raw":
        return y

    raise ValueError(f"Unknown TARGET_TRANSFORM: {TARGET_TRANSFORM}")


def log_auc(auc):
    return np.log1p(np.maximum(np.asarray(auc, dtype=float), AUC_MIN_POSITIVE))


def clean_abnormal_auc(df, problem_type):
    """
    Remove or clip abnormal AUC values.

    This function records an `auc_filter_reason` before dropping/clipping.
    It saves a separate report so you can inspect how many rows were affected.
    """
    df = df.copy()
    n0 = len(df)

    df["auc_mean"] = pd.to_numeric(df["auc_mean"], errors="coerce")
    df["auc_filter_reason"] = "kept"

    invalid = (~np.isfinite(df["auc_mean"])) | (df["auc_mean"] <= AUC_MIN_POSITIVE)
    df.loc[invalid, "auc_filter_reason"] = "non_finite_or_non_positive"

    # Global hard cap
    if AUC_ABS_MAX is not None:
        too_large_global = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > AUC_ABS_MAX)
        df.loc[too_large_global, "auc_filter_reason"] = f"above_global_cap_{AUC_ABS_MAX:.1e}"

    # Per-source upper quantile
    if AUC_SOURCE_UPPER_QUANTILE is not None:
        valid_for_q = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > AUC_MIN_POSITIVE)
        if valid_for_q.any():
            q = df.loc[valid_for_q, "auc_mean"].quantile(AUC_SOURCE_UPPER_QUANTILE)
            if np.isfinite(q):
                too_large_q = valid_for_q & (df["auc_mean"] > q)
                df.loc[too_large_q, "auc_filter_reason"] = f"above_source_q{AUC_SOURCE_UPPER_QUANTILE}"

    # Per-problem upper quantile
    if AUC_PROBLEM_UPPER_QUANTILE is not None:
        keys = ["problem_type", "fid", "iid", "dim"]
        for _, idx in df.groupby(keys).groups.items():
            sub = df.loc[idx]
            valid_for_q = np.isfinite(sub["auc_mean"]) & (sub["auc_mean"] > AUC_MIN_POSITIVE)
            if valid_for_q.sum() < 4:
                continue
            q = sub.loc[valid_for_q, "auc_mean"].quantile(AUC_PROBLEM_UPPER_QUANTILE)
            if np.isfinite(q):
                too_large_q = valid_for_q & (sub["auc_mean"] > q)
                df.loc[sub.index[too_large_q], "auc_filter_reason"] = f"above_problem_q{AUC_PROBLEM_UPPER_QUANTILE}"

    abnormal = df["auc_filter_reason"] != "kept"

    report = (
        df.groupby(["problem_type", "auc_filter_reason"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["problem_type", "auc_filter_reason"])
    )

    report_path = os.path.join(OUT_DIR, f"auc_filter_report_{problem_type}.csv")
    report.to_csv(report_path, index=False)

    abnormal_path = os.path.join(OUT_DIR, f"auc_abnormal_rows_{problem_type}.csv")
    df.loc[abnormal].to_csv(abnormal_path, index=False)

    if DROP_ABNORMAL_AUC:
        df = df.loc[~abnormal].copy()
        print(f"[AUC clean] {problem_type}: dropped {n0 - len(df)} abnormal AUC rows; kept {len(df)} / {n0}.")
    else:
        # Clip abnormal positive rows to finite caps instead of dropping.
        # Invalid/nonpositive rows are still dropped because log target cannot use them.
        df = df.loc[~invalid].copy()

        if AUC_ABS_MAX is not None:
            df["auc_mean"] = df["auc_mean"].clip(upper=AUC_ABS_MAX)

        if AUC_SOURCE_UPPER_QUANTILE is not None:
            q = df["auc_mean"].quantile(AUC_SOURCE_UPPER_QUANTILE)
            if np.isfinite(q):
                df["auc_mean"] = df["auc_mean"].clip(upper=q)

        print(f"[AUC clean] {problem_type}: clipped abnormal AUC rows; kept {len(df)} / {n0}.")

    return df


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

    df = clean_abnormal_auc(df, problem_type)

    df["auc_mean"] = pd.to_numeric(df["auc_mean"], errors="coerce").astype(float)
    df["target_auc"] = transform_target_auc(df["auc_mean"].to_numpy())
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
    for p in [
        BBOB_ELA_PATH, BBOB_PERF_PATH,
        MABBOB_ELA_PATH, MABBOB_PERF_PATH,
        LLM_ELA_PATH, LLM_PERF_PATH,
    ]:
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
# 4. Regressor training table
# ============================================================

def build_regressor_train_table(ela_df, perf_df):
    train_df_reg = pd.merge(
        ela_df,
        perf_df,
        on=problem_key_cols(),
        how="inner",
        suffixes=("", "_perf"),
    )
    return train_df_reg


def make_regressor_X(train_df_reg, feature_cols, all_algorithms=None):
    X_base = clean_X(train_df_reg[feature_cols])
    alg_dummies = pd.get_dummies(train_df_reg["algname"].astype(str), prefix="algname")

    X = pd.concat(
        [X_base.reset_index(drop=True), alg_dummies.reset_index(drop=True)],
        axis=1,
    )

    if all_algorithms is not None:
        all_alg_cols = [f"algname_{a}" for a in all_algorithms]
        for c in all_alg_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols + all_alg_cols]

    return X


# ============================================================
# 5. Metrics
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
    rows.append({
        "section": section,
        "source": source,
        "metric": metric,
        "value": value,
        "note": note,
    })


def train_fold_algorithm_mean_baseline(train_rows, test_rows):
    """
    Per-fold baseline for transformed AUC prediction:
    predict each algorithm's mean target_auc in the training fold.
    """
    global_mean = train_rows["target_auc"].mean()
    alg_mean = train_rows.groupby("algname")["target_auc"].mean().to_dict()
    return test_rows["algname"].map(alg_mean).fillna(global_mean).to_numpy(dtype=float)


def selector_validation_from_regressor_fold(test_rows, pred_target, train_rows, fold):
    """
    For each held-out problem instance:
      - predict all algorithms' transformed AUC;
      - choose algorithm with minimum predicted transformed AUC;
      - evaluate chosen algorithm by true AUC and true log-AUC.
    """
    tmp = test_rows.copy()
    tmp["pred_target_auc"] = pred_target
    tmp["pred_auc_backtransformed"] = inverse_transform_target(pred_target)
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
            fixed_actual_log_auc = float(log_auc(fixed_actual_auc))
        else:
            fixed_actual_auc = np.nan
            fixed_actual_log_auc = np.nan

        pred_actual_auc = float(pred_row["auc_mean"])
        oracle_actual_auc = float(oracle_row["auc_mean"])
        pred_actual_log_auc = float(log_auc(pred_actual_auc))
        oracle_actual_log_auc = float(log_auc(oracle_actual_auc))

        relative_regret = (
            pred_actual_auc / oracle_actual_auc - 1.0
            if oracle_actual_auc > AUC_MIN_POSITIVE and np.isfinite(pred_actual_auc) and np.isfinite(oracle_actual_auc)
            else np.nan
        )

        fixed_relative_regret = (
            fixed_actual_auc / oracle_actual_auc - 1.0
            if oracle_actual_auc > AUC_MIN_POSITIVE and np.isfinite(fixed_actual_auc) and np.isfinite(oracle_actual_auc)
            else np.nan
        )

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
            "pred_selected_actual_log_auc": pred_actual_log_auc,
            "oracle_actual_log_auc": oracle_actual_log_auc,
            "fixed_baseline_actual_log_auc": fixed_actual_log_auc,
            "raw_regret_vs_oracle": pred_actual_auc - oracle_actual_auc,
            "fixed_baseline_raw_regret_vs_oracle": fixed_actual_auc - oracle_actual_auc if np.isfinite(fixed_actual_auc) else np.nan,
            "log_regret_vs_oracle": pred_actual_log_auc - oracle_actual_log_auc,
            "fixed_baseline_log_regret_vs_oracle": fixed_actual_log_auc - oracle_actual_log_auc if np.isfinite(fixed_actual_log_auc) else np.nan,
            "relative_regret_vs_oracle": relative_regret,
            "fixed_baseline_relative_regret_vs_oracle": fixed_relative_regret,
            "selected_is_oracle": pred_row["algname"] == oracle_row["algname"],
            "n_algorithms_available": int(len(g)),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. Cross-validation
# ============================================================

def validate_regressor(train_df_reg, feature_cols):
    print("\n=== Log-AUC regressor validation ===")

    all_algorithms = sorted(train_df_reg["algname"].astype(str).unique().tolist())
    y = train_df_reg["target_auc"].astype(float).to_numpy()
    y_raw = train_df_reg["auc_mean"].astype(float).to_numpy()

    X = make_regressor_X(train_df_reg, feature_cols, all_algorithms=all_algorithms)
    groups = make_groups(train_df_reg)

    n_groups = len(np.unique(groups))
    n_splits = min(N_SPLITS, n_groups)

    if n_splits < 2:
        raise RuntimeError("Not enough groups for regressor GroupKFold validation.")

    reg_template = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features="log2",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    cv = GroupKFold(n_splits=n_splits)

    pred_target = np.full(len(train_df_reg), fill_value=np.nan, dtype=float)
    baseline_target = np.full(len(train_df_reg), fill_value=np.nan, dtype=float)
    fold_id = np.full(len(train_df_reg), fill_value=-1, dtype=int)
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

        selector_parts.append(
            selector_validation_from_regressor_fold(
                test_rows=test_rows,
                pred_target=fold_pred,
                train_rows=train_rows,
                fold=fold,
            )
        )

    pred_df = train_df_reg[problem_key_cols() + ["problem_name", "algname", "auc_mean", "target_auc"]].copy()
    pred_df["fold"] = fold_id
    pred_df["pred_target_auc"] = pred_target
    pred_df["baseline_alg_mean_pred_target_auc"] = baseline_target

    pred_df["pred_auc_backtransformed"] = inverse_transform_target(pred_target)
    pred_df["baseline_auc_backtransformed"] = inverse_transform_target(baseline_target)

    pred_df["abs_log_error"] = np.abs(pred_df["target_auc"] - pred_df["pred_target_auc"])
    pred_df["baseline_abs_log_error"] = np.abs(pred_df["target_auc"] - pred_df["baseline_alg_mean_pred_target_auc"])

    # Raw error is only for diagnostics. It can still be large, so do not use it as the primary metric.
    pred_df["abs_raw_error"] = np.abs(pred_df["auc_mean"] - pred_df["pred_auc_backtransformed"])
    pred_df["baseline_abs_raw_error"] = np.abs(pred_df["auc_mean"] - pred_df["baseline_auc_backtransformed"])

    pred_path = os.path.join(OUT_DIR, "validation_regressor_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved: {pred_path}")

    selector_df = pd.concat(selector_parts, ignore_index=True)
    selector_path = os.path.join(OUT_DIR, "validation_selector_from_regressor.csv")
    selector_df.to_csv(selector_path, index=False)
    print(f"Saved: {selector_path}")

    metric_rows = []

    # Main prediction metrics on transformed target.
    add_metric(metric_rows, "regressor_log_target", "r2", safe_r2(y, pred_target))
    add_metric(metric_rows, "regressor_log_target", "mae", float(mean_absolute_error(y, pred_target)))
    add_metric(metric_rows, "regressor_log_target", "rmse", rmse(y, pred_target))
    add_metric(metric_rows, "regressor_log_target", "spearman", safe_spearman(y, pred_target))
    add_metric(metric_rows, "regressor_log_target", "baseline_alg_mean_mae", float(mean_absolute_error(y, baseline_target)))
    add_metric(metric_rows, "regressor_log_target", "baseline_alg_mean_rmse", rmse(y, baseline_target))
    add_metric(
        metric_rows,
        "regressor_log_target",
        "mae_improvement_over_alg_mean_baseline",
        float(mean_absolute_error(y, baseline_target) - mean_absolute_error(y, pred_target)),
        note="positive means lower log-target MAE than algorithm-mean baseline",
    )

    # Diagnostics on raw scale. Not primary.
    raw_pred = inverse_transform_target(pred_target)
    raw_base = inverse_transform_target(baseline_target)
    add_metric(metric_rows, "regressor_raw_diagnostic", "mae", float(mean_absolute_error(y_raw, raw_pred)))
    add_metric(metric_rows, "regressor_raw_diagnostic", "baseline_alg_mean_mae", float(mean_absolute_error(y_raw, raw_base)))

    # Algorithm selection metrics.
    add_metric(metric_rows, "selector_from_regressor", "oracle_match_accuracy", float(selector_df["selected_is_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "mean_log_regret_vs_oracle", float(selector_df["log_regret_vs_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "median_log_regret_vs_oracle", float(selector_df["log_regret_vs_oracle"].median()))
    add_metric(metric_rows, "selector_from_regressor", "mean_fixed_baseline_log_regret_vs_oracle", float(selector_df["fixed_baseline_log_regret_vs_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "median_fixed_baseline_log_regret_vs_oracle", float(selector_df["fixed_baseline_log_regret_vs_oracle"].median()))
    add_metric(
        metric_rows,
        "selector_from_regressor",
        "mean_log_regret_improvement_over_fixed_baseline",
        float(selector_df["fixed_baseline_log_regret_vs_oracle"].mean() - selector_df["log_regret_vs_oracle"].mean()),
        note="positive means model selector has lower log-regret than fixed best-algorithm baseline",
    )
    add_metric(metric_rows, "selector_from_regressor", "mean_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].mean()))
    add_metric(metric_rows, "selector_from_regressor", "median_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].median()))

    for source, sub in pred_df.groupby("problem_type"):
        yt = sub["target_auc"].to_numpy(dtype=float)
        yp = sub["pred_target_auc"].to_numpy(dtype=float)
        yb = sub["baseline_alg_mean_pred_target_auc"].to_numpy(dtype=float)

        add_metric(metric_rows, "regressor_log_target", "r2", safe_r2(yt, yp), source=source)
        add_metric(metric_rows, "regressor_log_target", "mae", float(mean_absolute_error(yt, yp)), source=source)
        add_metric(metric_rows, "regressor_log_target", "rmse", rmse(yt, yp), source=source)
        add_metric(metric_rows, "regressor_log_target", "spearman", safe_spearman(yt, yp), source=source)
        add_metric(metric_rows, "regressor_log_target", "baseline_alg_mean_mae", float(mean_absolute_error(yt, yb)), source=source)
        add_metric(
            metric_rows,
            "regressor_log_target",
            "mae_improvement_over_alg_mean_baseline",
            float(mean_absolute_error(yt, yb) - mean_absolute_error(yt, yp)),
            source=source,
            note="positive means lower log-target MAE than algorithm-mean baseline",
        )

    for source, sub in selector_df.groupby("problem_type"):
        add_metric(metric_rows, "selector_from_regressor", "oracle_match_accuracy", float(sub["selected_is_oracle"].mean()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "mean_log_regret_vs_oracle", float(sub["log_regret_vs_oracle"].mean()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "median_log_regret_vs_oracle", float(sub["log_regret_vs_oracle"].median()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "mean_fixed_baseline_log_regret_vs_oracle", float(sub["fixed_baseline_log_regret_vs_oracle"].mean()), source=source)
        add_metric(
            metric_rows,
            "selector_from_regressor",
            "mean_log_regret_improvement_over_fixed_baseline",
            float(sub["fixed_baseline_log_regret_vs_oracle"].mean() - sub["log_regret_vs_oracle"].mean()),
            source=source,
            note="positive means model selector has lower log-regret than fixed best-algorithm baseline",
        )
        add_metric(metric_rows, "selector_from_regressor", "mean_relative_regret_vs_oracle", float(sub["relative_regret_vs_oracle"].mean()), source=source)
        add_metric(metric_rows, "selector_from_regressor", "median_relative_regret_vs_oracle", float(sub["relative_regret_vs_oracle"].median()), source=source)

    return reg_template, all_algorithms, X.columns.tolist(), pred_df, selector_df, pd.DataFrame(metric_rows)


# ============================================================
# 7. Plots
# ============================================================

def plot_log_pred_vs_true(reg_pred_df):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Color points by problem source so BBOB / MABBOB / LLM are distinguishable.
    source_order = ["LLM", "BBOB", "MABBOB"]
    colors = {
        "BBOB": "#1f77b4",
        "MABBOB": "#ff7f0e",
        "LLM": "#2ca02c",
    }

    plotted_sources = []
    for source in source_order:
        sub = reg_pred_df[reg_pred_df["problem_type"] == source]
        if sub.empty:
            continue

        ax.scatter(
            sub["target_auc"],
            sub["pred_target_auc"],
            s=12,
            alpha=0.42,
            color=colors.get(source, None),
            label=f"{source} (n={len(sub)})",
        )
        plotted_sources.append(source)

    # Plot any unexpected sources as fallback.
    for source in sorted(set(reg_pred_df["problem_type"].unique()) - set(plotted_sources)):
        sub = reg_pred_df[reg_pred_df["problem_type"] == source]
        ax.scatter(
            sub["target_auc"],
            sub["pred_target_auc"],
            s=12,
            alpha=0.42,
            label=f"{source} (n={len(sub)})",
        )

    x = reg_pred_df["target_auc"].to_numpy(dtype=float)
    y = reg_pred_df["pred_target_auc"].to_numpy(dtype=float)
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", label="Ideal prediction")
    ax.set_xlabel("True transformed AUC")
    ax.set_ylabel("Predicted transformed AUC")
    ax.set_title(f"Cross-validated regressor predictions ({TARGET_TRANSFORM} target)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "regressor_log_target_pred_vs_true.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_log_abs_error_by_source(reg_pred_df):
    sources = sorted(reg_pred_df["problem_type"].unique())
    data = [
        reg_pred_df.loc[reg_pred_df["problem_type"] == s, "abs_log_error"].dropna().values
        for s in sources
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=sources, showfliers=False)
    ax.set_ylabel("Absolute transformed-AUC prediction error")
    ax.set_title("Regressor transformed-target absolute error by source")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "regressor_log_abs_error_by_source.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_selector_log_regret_by_source(selector_df):
    sources = sorted(selector_df["problem_type"].unique())
    data_model = [
        selector_df.loc[selector_df["problem_type"] == s, "log_regret_vs_oracle"].dropna().values
        for s in sources
    ]
    data_base = [
        selector_df.loc[selector_df["problem_type"] == s, "fixed_baseline_log_regret_vs_oracle"].dropna().values
        for s in sources
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    positions_model = np.arange(len(sources)) * 2.0 + 1.0
    positions_base = positions_model + 0.7

    ax.boxplot(data_model, positions=positions_model, widths=0.55, showfliers=False)
    ax.boxplot(data_base, positions=positions_base, widths=0.55, showfliers=False)

    ax.set_xticks(positions_model + 0.35)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Log-regret vs oracle")
    ax.set_title("Algorithm-selection log-regret: regressor selector vs fixed baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.02,
        0.98,
        "Left box: regressor selector\nRight box: fixed best-algorithm baseline",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "selector_log_regret_by_source.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_selector_log_auc_by_source(selector_df):
    sources = sorted(selector_df["problem_type"].unique())

    data_oracle = [
        selector_df.loc[selector_df["problem_type"] == s, "oracle_actual_log_auc"].dropna().values
        for s in sources
    ]
    data_model = [
        selector_df.loc[selector_df["problem_type"] == s, "pred_selected_actual_log_auc"].dropna().values
        for s in sources
    ]
    data_base = [
        selector_df.loc[selector_df["problem_type"] == s, "fixed_baseline_actual_log_auc"].dropna().values
        for s in sources
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    positions_oracle = np.arange(len(sources)) * 3.0 + 1.0
    positions_model = positions_oracle + 0.7
    positions_base = positions_oracle + 1.4

    # Explicit colors for the three selector types.
    colors = {
        "Oracle": "#2ca02c",
        "Regressor selector": "#1f77b4",
        "Fixed baseline": "#ff7f0e",
    }

    def colored_boxplot(data, positions, color, label):
        ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor=color, edgecolor="black", alpha=0.55),
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )

        # Legend handle.
        ax.plot([], [], color=color, linewidth=8, alpha=0.55, label=label)

    colored_boxplot(data_oracle, positions_oracle, colors["Oracle"], "Best Algorithm per Problem")
    colored_boxplot(data_model, positions_model, colors["Regressor selector"], "Selected Best Algorithm per Problem")
    colored_boxplot(data_base, positions_base, colors["Fixed baseline"], "Best Algorithm Overall")

    ax.set_xticks(positions_oracle + 0.7)
    ax.set_xticklabels(sources)
    ax.set_ylabel("Actual log1p(AUC) of selected algorithm")
    ax.set_title("Actual log-AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "selector_actual_log_auc_by_source.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")



def plot_auc_distribution_after_filtering(train_df_reg):
    fig, ax = plt.subplots(figsize=(8, 5))

    for source, sub in train_df_reg.groupby("problem_type"):
        vals = log_auc(sub["auc_mean"].to_numpy())
        ax.hist(vals, bins=50, alpha=0.45, label=f"{source} (n={len(sub)})")

    ax.set_xlabel("log1p(AUC)")
    ax.set_ylabel("Count")
    ax.set_title("AUC distribution after abnormal-value filtering")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "auc_distribution_after_filtering_log_scale.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def make_plots(reg_pred_df, selector_df, train_df_reg):
    plot_auc_distribution_after_filtering(train_df_reg)
    plot_log_pred_vs_true(reg_pred_df)
    plot_log_abs_error_by_source(reg_pred_df)
    plot_selector_log_regret_by_source(selector_df)
    plot_selector_log_auc_by_source(selector_df)


# ============================================================
# 8. Final fit and save
# ============================================================

def final_fit_and_save(train_df_reg, feature_cols, reg_template, all_algorithms, reg_feature_cols, validation_meta):
    print("\n=== Final fit on all filtered data ===")

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
        "target_transform": TARGET_TRANSFORM,
        "auc_min_positive": AUC_MIN_POSITIVE,
        "inverse_transform_note": "For target_transform='log1p', use np.expm1(pred). Algorithm selection can use predicted transformed AUC directly because log1p is monotonic.",
        "meta": validation_meta,
    }

    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"Saved final log-AUC regressor AS model bundle to: {MODEL_SAVE_PATH}")


# ============================================================
# 9. Main
# ============================================================

def main():
    ela_df, perf_df = load_all_data()
    feature_cols = get_feature_cols(ela_df)

    print("\n=== Data summary after AUC cleaning ===")
    print(f"ELA rows: {len(ela_df)}")
    print(f"Performance rows: {len(perf_df)}")
    print(f"Feature cols: {len(feature_cols)}")
    print("\nELA source counts:")
    print(ela_df["problem_type"].value_counts())
    print("\nPerformance source counts after AUC cleaning:")
    print(perf_df["problem_type"].value_counts())

    if len(feature_cols) == 0:
        raise RuntimeError("No usable numeric ELA features after cleaning.")

    train_df_reg = build_regressor_train_table(ela_df, perf_df)

    print("\n=== Matched regressor training table ===")
    print(f"Regressor rows: {len(train_df_reg)}")
    print("\nRegressor rows by source:")
    print(train_df_reg["problem_type"].value_counts())

    train_df_reg[problem_key_cols() + ["problem_name", "algname", "auc_mean", "target_auc"]].to_csv(
        os.path.join(OUT_DIR, "training_regressor_table_keys.csv"),
        index=False,
    )

    # Save summary of target scale for quick inspection.
    target_summary = (
        train_df_reg
        .assign(log_auc=lambda d: log_auc(d["auc_mean"].to_numpy()))
        .groupby("problem_type")
        .agg(
            n_rows=("auc_mean", "count"),
            auc_min=("auc_mean", "min"),
            auc_median=("auc_mean", "median"),
            auc_mean=("auc_mean", "mean"),
            auc_max=("auc_mean", "max"),
            log_auc_min=("log_auc", "min"),
            log_auc_median=("log_auc", "median"),
            log_auc_mean=("log_auc", "mean"),
            log_auc_max=("log_auc", "max"),
        )
        .reset_index()
    )
    target_summary_path = os.path.join(OUT_DIR, "target_auc_summary_after_filtering.csv")
    target_summary.to_csv(target_summary_path, index=False)
    print(f"Saved: {target_summary_path}")

    reg_template, all_algorithms, reg_feature_cols, reg_pred_df, selector_df, metrics = validate_regressor(
        train_df_reg,
        feature_cols,
    )

    metrics_path = os.path.join(OUT_DIR, "validation_metrics_summary.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    make_plots(reg_pred_df, selector_df, train_df_reg)

    validation_meta = {
        "model_type": "regressor_only_algorithm_selection",
        "target_transform": TARGET_TRANSFORM,
        "auc_cleaning": {
            "auc_min_positive": AUC_MIN_POSITIVE,
            "auc_abs_max": AUC_ABS_MAX,
            "auc_source_upper_quantile": AUC_SOURCE_UPPER_QUANTILE,
            "auc_problem_upper_quantile": AUC_PROBLEM_UPPER_QUANTILE,
            "drop_abnormal_auc": DROP_ABNORMAL_AUC,
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
        final_fit_and_save(
            train_df_reg=train_df_reg,
            feature_cols=feature_cols,
            reg_template=reg_template,
            all_algorithms=all_algorithms,
            reg_feature_cols=reg_feature_cols,
            validation_meta=validation_meta,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
