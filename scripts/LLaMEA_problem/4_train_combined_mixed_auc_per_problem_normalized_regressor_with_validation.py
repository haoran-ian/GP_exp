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

BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
BBOB_PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"
MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
MABBOB_PERF_PATH = "data/MABBOB/mabbob_algorithm_auc_performance.csv"
LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"
LLM_PERF_PATH = "data/LLM/llm_algorithm_auc_performance.csv"

MODEL_SAVE_PATH = "data/Combined/models/bbob_mabbob_llm_mixed_auc_per_problem_normalized_regressor_as_model.joblib"
OUT_DIR = "data/Combined/validation_mixed_auc_per_problem_normalized_regressor"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_SPLITS = 5
N_ESTIMATORS = 500
SAVE_FINAL_MODEL = True

# BBOB/MABBOB: raw AUC. LLM: abnormal filtering + log1p(AUC).
AUC_MIN_POSITIVE = 1e-300
LLM_AUC_ABS_MAX = 1e100
LLM_AUC_UPPER_QUANTILE = 0.995
DROP_LLM_ABNORMAL_AUC = True

# Now normalization is fitted for each individual problem instance.
# Problem identity: problem_type + fid + iid + dim.
NORMALIZATION_METHOD = "minmax"  # minmax, zscore, robust
NORMALIZATION_EPS = 1e-12

META_COLS = {
    "problem_type", "problem_name", "fid", "iid", "dim", "seed", "n_samples",
    "instance_id", "mabbob_instance_id", "llm_problem_id", "selection_method",
    "source_dataset", "lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
}


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
        print(f"[Clean] Dropped {n0-len(df)} invalid {problem_type} {stage} rows; kept {len(df)} / {n0}.")
    return df


def ensure_problem_keys(df, problem_type):
    df = harmonize_feature_names(df.copy())
    df = drop_invalid_problem_rows(df, problem_type, "ELA")
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
        ids = pd.to_numeric(df["mabbob_instance_id"], errors="coerce")
        df = df.loc[np.isfinite(ids)].copy()
        df["mabbob_instance_id"] = ids.loc[df.index].astype(int)
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
        ids = pd.to_numeric(df["llm_problem_id"], errors="coerce")
        n0 = len(df)
        df = df.loc[np.isfinite(ids)].copy()
        df["llm_problem_id"] = ids.loc[df.index].astype(int)
        if len(df) < n0:
            print(f"[Clean] Dropped {n0-len(df)} LLM ELA rows with invalid llm_problem_id.")
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
        df.loc[invalid].to_csv(os.path.join(OUT_DIR, f"auc_nonfinite_rows_{problem_type}.csv"), index=False)
        df = df.loc[~invalid].copy()
        df["auc_transform"] = "raw"
        df["transformed_auc"] = df["auc_mean"].astype(float)
        report = pd.DataFrame([
            {"problem_type": problem_type, "auc_filter_reason": "kept_finite_raw_auc", "n_rows": len(df)},
            {"problem_type": problem_type, "auc_filter_reason": "dropped_nonfinite_auc", "n_rows": int(invalid.sum())},
        ])
        report.to_csv(os.path.join(OUT_DIR, f"auc_mixed_filter_report_{problem_type}.csv"), index=False)
        print(f"[AUC clean] {problem_type}: no abnormal finite-AUC filtering; kept {len(df)} / {n0}.")
        return df
    if problem_type == "LLM":
        invalid = (~np.isfinite(df["auc_mean"])) | (df["auc_mean"] <= AUC_MIN_POSITIVE)
        df.loc[invalid, "auc_filter_reason"] = "non_finite_or_non_positive"
        if LLM_AUC_ABS_MAX is not None:
            too_large = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > LLM_AUC_ABS_MAX)
            df.loc[too_large, "auc_filter_reason"] = f"above_llm_global_cap_{LLM_AUC_ABS_MAX:.1e}"
        if LLM_AUC_UPPER_QUANTILE is not None:
            valid_for_q = np.isfinite(df["auc_mean"]) & (df["auc_mean"] > AUC_MIN_POSITIVE)
            if valid_for_q.any():
                q = df.loc[valid_for_q, "auc_mean"].quantile(LLM_AUC_UPPER_QUANTILE)
                if np.isfinite(q):
                    too_large_q = valid_for_q & (df["auc_mean"] > q)
                    df.loc[too_large_q, "auc_filter_reason"] = f"above_llm_q{LLM_AUC_UPPER_QUANTILE}"
        abnormal = df["auc_filter_reason"] != "kept"
        report = df.groupby(["problem_type", "auc_filter_reason"]).size().reset_index(name="n_rows")
        report.to_csv(os.path.join(OUT_DIR, f"auc_mixed_filter_report_{problem_type}.csv"), index=False)
        df.loc[abnormal].to_csv(os.path.join(OUT_DIR, f"auc_abnormal_rows_{problem_type}.csv"), index=False)
        if DROP_LLM_ABNORMAL_AUC:
            df = df.loc[~abnormal].copy()
            print(f"[AUC clean] {problem_type}: dropped {n0-len(df)} abnormal AUC rows; kept {len(df)} / {n0}.")
        else:
            df = df.loc[~invalid].copy()
            if LLM_AUC_ABS_MAX is not None:
                df["auc_mean"] = df["auc_mean"].clip(upper=LLM_AUC_ABS_MAX)
            if LLM_AUC_UPPER_QUANTILE is not None:
                q = df["auc_mean"].quantile(LLM_AUC_UPPER_QUANTILE)
                if np.isfinite(q):
                    df["auc_mean"] = df["auc_mean"].clip(upper=q)
            print(f"[AUC clean] {problem_type}: clipped abnormal AUC rows; kept {len(df)} / {n0}.")
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
        ids = pd.to_numeric(df["mabbob_instance_id"], errors="coerce")
        df = df.loc[np.isfinite(ids)].copy()
        df["mabbob_instance_id"] = ids.loc[df.index].astype(int)
        df["fid"] = -100
        df["iid"] = df["mabbob_instance_id"].astype(int)
        df["problem_name"] = df["iid"].apply(lambda x: f"MABBOB_{int(x)}")
    elif problem_type == "LLM":
        if "llm_problem_id" not in df.columns:
            if "iid" in df.columns:
                df["llm_problem_id"] = df["iid"]
            else:
                raise ValueError("LLM performance must contain llm_problem_id or iid.")
        ids = pd.to_numeric(df["llm_problem_id"], errors="coerce")
        n0 = len(df)
        df = df.loc[np.isfinite(ids)].copy()
        df["llm_problem_id"] = ids.loc[df.index].astype(int)
        if len(df) < n0:
            print(f"[Clean] Dropped {n0-len(df)} LLM performance rows with invalid llm_problem_id.")
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
    feature_cols = []
    for c in ela_df.columns:
        if c in META_COLS or c in ["FAILED", "ERROR"] or c.endswith(".FAILED") or c.endswith(".ERROR"):
            continue
        if pd.api.types.is_numeric_dtype(ela_df[c]):
            feature_cols.append(c)
    X = ela_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in feature_cols if not X[c].isna().all()]
    if feature_cols:
        nunique = X[feature_cols].nunique(dropna=True)
        feature_cols = [c for c in feature_cols if nunique[c] > 1]
    return sorted(feature_cols)


def make_groups(df):
    out = []
    for _, r in df.iterrows():
        if r["problem_type"] == "BBOB":
            out.append(f"BBOB_F{int(r['fid'])}")
        elif r["problem_type"] == "MABBOB":
            out.append(f"MABBOB_{int(r['iid'])}")
        elif r["problem_type"] == "LLM":
            out.append(f"LLM_{int(r['iid'])}")
        else:
            out.append(f"{r['problem_type']}_{int(r['fid'])}_{int(r['iid'])}")
    return np.asarray(out)


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


def fit_problem_normalizers(perf_df):
    df = perf_df.copy()
    df["problem_key"] = make_problem_key(df)
    params = {}
    for key, sub in df.groupby("problem_key"):
        values = sub["transformed_auc"].to_numpy(float)
        if NORMALIZATION_METHOD == "minmax":
            vmin, vmax = float(np.min(values)), float(np.max(values))
            scale = vmax - vmin
            if not np.isfinite(scale) or scale <= NORMALIZATION_EPS:
                scale = 1.0
            params[key] = {"method": "minmax", "min": vmin, "max": vmax, "scale": scale}
        elif NORMALIZATION_METHOD == "zscore":
            mean, std = float(np.mean(values)), float(np.std(values))
            if not np.isfinite(std) or std <= NORMALIZATION_EPS:
                std = 1.0
            params[key] = {"method": "zscore", "mean": mean, "std": std}
        elif NORMALIZATION_METHOD == "robust":
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            iqr = float(q75 - q25)
            if not np.isfinite(iqr) or iqr <= NORMALIZATION_EPS:
                iqr = 1.0
            params[key] = {"method": "robust", "median": median, "q25": float(q25), "q75": float(q75), "iqr": iqr}
        else:
            raise ValueError(NORMALIZATION_METHOD)
    return params


def transform_value_by_problem(value, key, params):
    p = params[key]
    value = np.asarray(value, dtype=float)
    if p["method"] == "minmax":
        return (value - p["min"]) / p["scale"]
    if p["method"] == "zscore":
        return (value - p["mean"]) / p["std"]
    if p["method"] == "robust":
        return (value - p["median"]) / p["iqr"]
    raise ValueError(p["method"])


def inverse_transform_value_by_problem(target, key, params):
    p = params[key]
    target = np.asarray(target, dtype=float)
    if p["method"] == "minmax":
        return target * p["scale"] + p["min"]
    if p["method"] == "zscore":
        return target * p["std"] + p["mean"]
    if p["method"] == "robust":
        return target * p["iqr"] + p["median"]
    raise ValueError(p["method"])


def add_problem_normalized_target(perf_df, params):
    perf_df = perf_df.copy()
    perf_df["problem_key"] = make_problem_key(perf_df)
    perf_df["target_auc"] = np.nan
    for key, idx in perf_df.groupby("problem_key").groups.items():
        perf_df.loc[idx, "target_auc"] = transform_value_by_problem(
            perf_df.loc[idx, "transformed_auc"].to_numpy(float), key, params
        )
    return perf_df


def build_regressor_train_table(ela_df, perf_df):
    out = pd.merge(ela_df, perf_df, on=problem_key_cols(), how="inner", suffixes=("", "_perf"))
    out["problem_key"] = make_problem_key(out)
    return out


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


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    return np.nan if len(np.unique(y_true)) <= 1 else float(r2_score(y_true, y_pred))


def safe_spearman(y_true, y_pred):
    return np.nan if len(y_true) < 3 else float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def add_metric(rows, section, metric, value, source="ALL", note=""):
    rows.append({"section": section, "source": source, "metric": metric, "value": value, "note": note})


def train_fold_algorithm_mean_baseline(train_rows, test_rows):
    global_mean = train_rows["target_auc"].mean()
    alg_mean = train_rows.groupby("algname")["target_auc"].mean().to_dict()
    return test_rows["algname"].map(alg_mean).fillna(global_mean).to_numpy(float)


def selector_validation_from_regressor_fold(test_rows, pred_target, train_rows, fold):
    tmp = test_rows.copy()
    tmp["pred_target_auc"] = pred_target
    rows = []
    fixed_alg = train_rows.groupby("algname")["target_auc"].mean().idxmin()
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
        rows.append({
            "fold": int(fold), "problem_type": key[0], "fid": int(key[1]), "iid": int(key[2]), "dim": int(key[3]),
            "problem_key": pred_row["problem_key"], "pred_selected_alg": pred_row["algname"], "oracle_best_alg": oracle_row["algname"],
            "fixed_baseline_alg": fixed_alg,
            "pred_selected_actual_auc": pred_actual_auc, "oracle_actual_auc": oracle_actual_auc,
            "fixed_baseline_actual_auc": fixed_actual_auc,
            "pred_selected_actual_target_auc": pred_actual_target, "oracle_actual_target_auc": oracle_actual_target,
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


def validate_regressor(train_df_reg, feature_cols, problem_normalizer_params):
    print("\n=== Mixed-transform per-problem-normalized AUC regressor validation ===")
    all_algorithms = sorted(train_df_reg["algname"].astype(str).unique())
    y = train_df_reg["target_auc"].astype(float).to_numpy()
    y_raw = train_df_reg["auc_mean"].astype(float).to_numpy()
    X = make_regressor_X(train_df_reg, feature_cols, all_algorithms=all_algorithms)
    groups = make_groups(train_df_reg)
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for GroupKFold validation.")
    reg_template = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features="log2", n_jobs=-1, random_state=RANDOM_SEED)
    cv = GroupKFold(n_splits=n_splits)
    pred_target = np.full(len(train_df_reg), np.nan)
    baseline_target = np.full(len(train_df_reg), np.nan)
    fold_id = np.full(len(train_df_reg), -1, dtype=int)
    selector_parts = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"[Regressor CV] fold {fold+1}/{n_splits}: train={len(train_idx)}, test={len(test_idx)}")
        reg = clone(reg_template)
        reg.fit(X.iloc[train_idx], y[train_idx])
        fold_pred = reg.predict(X.iloc[test_idx])
        pred_target[test_idx] = fold_pred
        train_rows = train_df_reg.iloc[train_idx].copy()
        test_rows = train_df_reg.iloc[test_idx].copy()
        baseline_target[test_idx] = train_fold_algorithm_mean_baseline(train_rows, test_rows)
        fold_id[test_idx] = fold
        selector_parts.append(selector_validation_from_regressor_fold(test_rows, fold_pred, train_rows, fold))
    pred_df = train_df_reg[problem_key_cols() + ["problem_key", "problem_name", "algname", "auc_mean", "auc_transform", "transformed_auc", "target_auc"]].copy()
    pred_df["fold"] = fold_id
    pred_df["pred_target_auc"] = pred_target
    pred_df["baseline_alg_mean_pred_target_auc"] = baseline_target
    pred_df["pred_transformed_auc_backtransformed"] = np.nan
    pred_df["baseline_transformed_auc_backtransformed"] = np.nan
    for key, idx in pred_df.groupby("problem_key").groups.items():
        pred_df.loc[idx, "pred_transformed_auc_backtransformed"] = inverse_transform_value_by_problem(pred_df.loc[idx, "pred_target_auc"].to_numpy(float), key, problem_normalizer_params)
        pred_df.loc[idx, "baseline_transformed_auc_backtransformed"] = inverse_transform_value_by_problem(pred_df.loc[idx, "baseline_alg_mean_pred_target_auc"].to_numpy(float), key, problem_normalizer_params)
    pred_df["pred_raw_auc_backtransformed"] = np.nan
    pred_df["baseline_raw_auc_backtransformed"] = np.nan
    for source, idx in pred_df.groupby("problem_type").groups.items():
        pred_t = pred_df.loc[idx, "pred_transformed_auc_backtransformed"].to_numpy(float)
        base_t = pred_df.loc[idx, "baseline_transformed_auc_backtransformed"].to_numpy(float)
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
    rows = []
    add_metric(rows, "regressor_per_problem_normalized_target", "r2", safe_r2(y, pred_target))
    add_metric(rows, "regressor_per_problem_normalized_target", "mae", float(mean_absolute_error(y, pred_target)))
    add_metric(rows, "regressor_per_problem_normalized_target", "rmse", rmse(y, pred_target))
    add_metric(rows, "regressor_per_problem_normalized_target", "spearman", safe_spearman(y, pred_target))
    add_metric(rows, "regressor_per_problem_normalized_target", "baseline_alg_mean_mae", float(mean_absolute_error(y, baseline_target)))
    add_metric(rows, "regressor_per_problem_normalized_target", "baseline_alg_mean_rmse", rmse(y, baseline_target))
    add_metric(rows, "regressor_per_problem_normalized_target", "mae_improvement_over_alg_mean_baseline", float(mean_absolute_error(y, baseline_target) - mean_absolute_error(y, pred_target)), note="positive means lower normalized-target MAE than algorithm-mean baseline")
    add_metric(rows, "regressor_raw_diagnostic", "mae", float(mean_absolute_error(y_raw, pred_df["pred_raw_auc_backtransformed"])))
    add_metric(rows, "regressor_raw_diagnostic", "baseline_alg_mean_mae", float(mean_absolute_error(y_raw, pred_df["baseline_raw_auc_backtransformed"])))
    add_metric(rows, "selector_from_regressor", "oracle_match_accuracy", float(selector_df["selected_is_oracle"].mean()))
    add_metric(rows, "selector_from_regressor", "mean_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].mean()))
    add_metric(rows, "selector_from_regressor", "median_normalized_regret_vs_oracle", float(selector_df["normalized_regret_vs_oracle"].median()))
    add_metric(rows, "selector_from_regressor", "mean_fixed_baseline_normalized_regret_vs_oracle", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean()))
    add_metric(rows, "selector_from_regressor", "mean_normalized_regret_improvement_over_fixed_baseline", float(selector_df["fixed_baseline_normalized_regret_vs_oracle"].mean() - selector_df["normalized_regret_vs_oracle"].mean()), note="positive means model selector has lower normalized-regret than fixed baseline")
    add_metric(rows, "selector_from_regressor", "mean_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].mean()))
    add_metric(rows, "selector_from_regressor", "median_relative_regret_vs_oracle", float(selector_df["relative_regret_vs_oracle"].median()))
    for source, sub in pred_df.groupby("problem_type"):
        yt, yp, yb = sub["target_auc"].to_numpy(float), sub["pred_target_auc"].to_numpy(float), sub["baseline_alg_mean_pred_target_auc"].to_numpy(float)
        add_metric(rows, "regressor_per_problem_normalized_target", "r2", safe_r2(yt, yp), source=source)
        add_metric(rows, "regressor_per_problem_normalized_target", "mae", float(mean_absolute_error(yt, yp)), source=source)
        add_metric(rows, "regressor_per_problem_normalized_target", "rmse", rmse(yt, yp), source=source)
        add_metric(rows, "regressor_per_problem_normalized_target", "spearman", safe_spearman(yt, yp), source=source)
        add_metric(rows, "regressor_per_problem_normalized_target", "baseline_alg_mean_mae", float(mean_absolute_error(yt, yb)), source=source)
        add_metric(rows, "regressor_per_problem_normalized_target", "mae_improvement_over_alg_mean_baseline", float(mean_absolute_error(yt, yb) - mean_absolute_error(yt, yp)), source=source, note="positive means lower normalized-target MAE than algorithm-mean baseline")
    for source, sub in selector_df.groupby("problem_type"):
        add_metric(rows, "selector_from_regressor", "oracle_match_accuracy", float(sub["selected_is_oracle"].mean()), source=source)
        add_metric(rows, "selector_from_regressor", "mean_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].mean()), source=source)
        add_metric(rows, "selector_from_regressor", "median_normalized_regret_vs_oracle", float(sub["normalized_regret_vs_oracle"].median()), source=source)
        add_metric(rows, "selector_from_regressor", "mean_fixed_baseline_normalized_regret_vs_oracle", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean()), source=source)
        add_metric(rows, "selector_from_regressor", "mean_normalized_regret_improvement_over_fixed_baseline", float(sub["fixed_baseline_normalized_regret_vs_oracle"].mean() - sub["normalized_regret_vs_oracle"].mean()), source=source, note="positive means model selector has lower normalized-regret than fixed baseline")
        add_metric(rows, "selector_from_regressor", "mean_relative_regret_vs_oracle", float(sub["relative_regret_vs_oracle"].mean()), source=source)
        add_metric(rows, "selector_from_regressor", "median_relative_regret_vs_oracle", float(sub["relative_regret_vs_oracle"].median()), source=source)
    return reg_template, all_algorithms, X.columns.tolist(), pred_df, selector_df, pd.DataFrame(rows)


def plot_pred_vs_true(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"BBOB": "#1f77b4", "MABBOB": "#ff7f0e", "LLM": "#2ca02c"}
    for source in ["BBOB", "MABBOB", "LLM"]:
        sub = df[df["problem_type"] == source]
        if len(sub):
            ax.scatter(sub["target_auc"], sub["pred_target_auc"], s=12, alpha=0.42, color=colors[source], label=f"{source} (n={len(sub)})")
    x, y = df["target_auc"].to_numpy(float), df["pred_target_auc"].to_numpy(float)
    lo, hi = min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2, color="black", label="Ideal")
    ax.set_xlabel("True per-problem-normalized target")
    ax.set_ylabel("Predicted per-problem-normalized target")
    ax.set_title(f"Cross-validated predictions ({NORMALIZATION_METHOD}, per-problem)")
    ax.grid(alpha=0.25); ax.legend(loc="best", frameon=True); plt.tight_layout()
    out = os.path.join(PLOT_DIR, "regressor_per_problem_normalized_target_pred_vs_true.png")
    plt.savefig(out, dpi=300); plt.close(); print(f"Saved: {out}")


def plot_abs_error_by_source(df):
    sources = sorted(df["problem_type"].unique())
    data = [df.loc[df["problem_type"] == s, "abs_normalized_error"].dropna().values for s in sources]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=sources, showfliers=False)
    ax.set_ylabel("Absolute per-problem-normalized prediction error")
    ax.set_title("Regressor absolute error by source")
    ax.grid(axis="y", alpha=0.25); plt.tight_layout()
    out = os.path.join(PLOT_DIR, "regressor_per_problem_normalized_abs_error_by_source.png")
    plt.savefig(out, dpi=300); plt.close(); print(f"Saved: {out}")


def plot_selector_auc_by_source(selector_df):
    sources = sorted(selector_df["problem_type"].unique())
    data_oracle = [selector_df.loc[selector_df["problem_type"] == s, "oracle_actual_target_auc"].dropna().values for s in sources]
    data_model = [selector_df.loc[selector_df["problem_type"] == s, "pred_selected_actual_target_auc"].dropna().values for s in sources]
    data_base = [selector_df.loc[selector_df["problem_type"] == s, "fixed_baseline_actual_target_auc"].dropna().values for s in sources]
    fig, ax = plt.subplots(figsize=(10, 5))
    pos_o = np.arange(len(sources)) * 3.0 + 1.0
    pos_m, pos_b = pos_o + 0.7, pos_o + 1.4
    def bp(data, pos, color, label):
        ax.boxplot(data, positions=pos, widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=color, edgecolor="black", alpha=0.55),
                   medianprops=dict(color="black", linewidth=1.4),
                   whiskerprops=dict(color="black"), capprops=dict(color="black"))
        ax.plot([], [], color=color, linewidth=8, alpha=0.55, label=label)
    bp(data_oracle, pos_o, "#2ca02c", "Oracle")
    bp(data_model, pos_m, "#1f77b4", "Regressor selector")
    bp(data_base, pos_b, "#ff7f0e", "Fixed baseline")
    ax.set_xticks(pos_o + 0.7); ax.set_xticklabels(sources)
    ax.set_ylabel("Actual per-problem-normalized target of selected algorithm")
    ax.set_title("Actual target: oracle vs regressor selector vs fixed baseline")
    ax.grid(axis="y", alpha=0.25); ax.legend(loc="upper right", frameon=True); plt.tight_layout()
    out = os.path.join(PLOT_DIR, "selector_actual_per_problem_normalized_auc_by_source.png")
    plt.savefig(out, dpi=300); plt.close(); print(f"Saved: {out}")


def plot_selector_regret_by_source(selector_df):
    sources = sorted(selector_df["problem_type"].unique())
    data_model = [selector_df.loc[selector_df["problem_type"] == s, "normalized_regret_vs_oracle"].dropna().values for s in sources]
    data_base = [selector_df.loc[selector_df["problem_type"] == s, "fixed_baseline_normalized_regret_vs_oracle"].dropna().values for s in sources]
    fig, ax = plt.subplots(figsize=(9, 5))
    pos_m = np.arange(len(sources)) * 2.0 + 1.0
    pos_b = pos_m + 0.7
    bm = ax.boxplot(data_model, positions=pos_m, widths=0.55, showfliers=False, patch_artist=True)
    bb = ax.boxplot(data_base, positions=pos_b, widths=0.55, showfliers=False, patch_artist=True)
    for b in bm["boxes"]: b.set(facecolor="#1f77b4", alpha=0.55)
    for b in bb["boxes"]: b.set(facecolor="#ff7f0e", alpha=0.55)
    ax.plot([], [], color="#1f77b4", linewidth=8, alpha=0.55, label="Regressor selector")
    ax.plot([], [], color="#ff7f0e", linewidth=8, alpha=0.55, label="Fixed baseline")
    ax.set_xticks(pos_m + 0.35); ax.set_xticklabels(sources)
    ax.set_ylabel("Per-problem-normalized regret vs oracle")
    ax.set_title("Algorithm-selection regret")
    ax.grid(axis="y", alpha=0.25); ax.legend(loc="upper right", frameon=True); plt.tight_layout()
    out = os.path.join(PLOT_DIR, "selector_per_problem_normalized_regret_by_source.png")
    plt.savefig(out, dpi=300); plt.close(); print(f"Saved: {out}")


def plot_target_distribution(train_df_reg):
    fig, ax = plt.subplots(figsize=(8, 5))
    for source, sub in train_df_reg.groupby("problem_type"):
        ax.hist(sub["target_auc"].to_numpy(float), bins=50, alpha=0.45, label=f"{source} (n={len(sub)})")
    ax.set_xlabel("Per-problem-normalized target")
    ax.set_ylabel("Count")
    ax.set_title("Target distribution after per-problem normalization")
    ax.legend(); ax.grid(axis="y", alpha=0.25); plt.tight_layout()
    out = os.path.join(PLOT_DIR, "target_distribution_per_problem_normalized.png")
    plt.savefig(out, dpi=300); plt.close(); print(f"Saved: {out}")


def make_plots(pred_df, selector_df, train_df_reg):
    plot_target_distribution(train_df_reg)
    plot_pred_vs_true(pred_df)
    plot_abs_error_by_source(pred_df)
    plot_selector_regret_by_source(selector_df)
    plot_selector_auc_by_source(selector_df)


def final_fit_and_save(train_df_reg, feature_cols, reg_template, all_algorithms, reg_feature_cols, validation_meta, problem_normalizer_params):
    print("\n=== Final fit on all mixed-transform per-problem-normalized data ===")
    X_reg = make_regressor_X(train_df_reg, feature_cols, all_algorithms=all_algorithms)[reg_feature_cols]
    y_reg = train_df_reg["target_auc"].astype(float).values
    reg = clone(reg_template)
    reg.fit(X_reg, y_reg)
    bundle = {
        "regressor": reg,
        "feature_cols": feature_cols,
        "reg_feature_cols": reg_feature_cols,
        "algorithms": all_algorithms,
        "target_transform": "mixed_raw_log_per_problem_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_level": "problem_instance",
        "problem_key_definition": problem_key_cols(),
        "problem_normalizer_params": problem_normalizer_params,
        "auc_min_positive": AUC_MIN_POSITIVE,
        "inverse_transform_note": "Predictions are per-problem-normalized targets. For new unseen problems, raw-AUC inverse transform is unavailable unless a per-problem normalizer is fitted from known algorithm performances. For algorithm selection, use argmin predicted target.",
        "meta": validation_meta,
    }
    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"Saved final model bundle to: {MODEL_SAVE_PATH}")


def main():
    ela_df, perf_df = load_all_data()
    problem_normalizer_params = fit_problem_normalizers(perf_df)
    perf_df = add_problem_normalized_target(perf_df, problem_normalizer_params)
    feature_cols = get_feature_cols(ela_df)
    print("\n=== Data summary after mixed AUC transform and per-problem normalization ===")
    print(f"ELA rows: {len(ela_df)}")
    print(f"Performance rows: {len(perf_df)}")
    print(f"Feature cols: {len(feature_cols)}")
    print(f"Normalization method: {NORMALIZATION_METHOD}")
    print(f"Number of problem normalizers: {len(problem_normalizer_params)}")
    print("\nELA source counts:"); print(ela_df["problem_type"].value_counts())
    print("\nPerformance source counts:"); print(perf_df["problem_type"].value_counts())
    if not feature_cols:
        raise RuntimeError("No usable numeric ELA features after cleaning.")
    normalizer_path = os.path.join(OUT_DIR, "per_problem_auc_normalizer_params.json")
    with open(normalizer_path, "w") as f: json.dump(problem_normalizer_params, f, indent=2)
    train_df_reg = build_regressor_train_table(ela_df, perf_df)
    print("\n=== Matched regressor training table ===")
    print(f"Regressor rows: {len(train_df_reg)}")
    print("\nRegressor rows by source:"); print(train_df_reg["problem_type"].value_counts())
    train_df_reg[problem_key_cols() + ["problem_key", "problem_name", "algname", "auc_mean", "auc_transform", "transformed_auc", "target_auc"]].to_csv(os.path.join(OUT_DIR, "training_regressor_table_keys.csv"), index=False)
    target_summary = train_df_reg.groupby("problem_type").agg(
        n_rows=("auc_mean", "count"), n_problems=("problem_key", "nunique"),
        auc_min=("auc_mean", "min"), auc_median=("auc_mean", "median"), auc_mean=("auc_mean", "mean"), auc_max=("auc_mean", "max"),
        transformed_min=("transformed_auc", "min"), transformed_median=("transformed_auc", "median"), transformed_mean=("transformed_auc", "mean"), transformed_max=("transformed_auc", "max"),
        target_min=("target_auc", "min"), target_median=("target_auc", "median"), target_mean=("target_auc", "mean"), target_max=("target_auc", "max"),
    ).reset_index()
    target_summary_path = os.path.join(OUT_DIR, "target_auc_summary_after_per_problem_normalization.csv")
    target_summary.to_csv(target_summary_path, index=False)
    reg_template, all_algorithms, reg_feature_cols, pred_df, selector_df, metrics = validate_regressor(train_df_reg, feature_cols, problem_normalizer_params)
    metrics_path = os.path.join(OUT_DIR, "validation_metrics_summary.csv")
    metrics.to_csv(metrics_path, index=False)
    make_plots(pred_df, selector_df, train_df_reg)
    validation_meta = {
        "model_type": "regressor_only_algorithm_selection",
        "target_transform": "mixed_raw_log_per_problem_normalized_auc",
        "normalization_method": NORMALIZATION_METHOD,
        "normalization_level": "problem_instance",
        "problem_key_definition": problem_key_cols(),
        "problem_normalizer_params_file": normalizer_path,
        "auc_handling": {
            "BBOB": {"transform": "raw", "abnormal_auc_filtering": False, "drop_nonfinite_auc_only": True},
            "MABBOB": {"transform": "raw", "abnormal_auc_filtering": False, "drop_nonfinite_auc_only": True},
            "LLM": {"transform": "log1p", "abnormal_auc_filtering": True, "llm_auc_abs_max": LLM_AUC_ABS_MAX, "llm_auc_upper_quantile": LLM_AUC_UPPER_QUANTILE, "drop_llm_abnormal_auc": DROP_LLM_ABNORMAL_AUC},
        },
        "train_sources": ["BBOB", "MABBOB", "LLM"],
        "regressor_rows": int(len(train_df_reg)),
        "n_problem_normalizers": int(len(problem_normalizer_params)),
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
            validation_meta[f"{row['section']}.{row['metric']}"] = None if pd.isna(row["value"]) else float(row["value"])
    with open(os.path.join(OUT_DIR, "validation_meta.json"), "w") as f: json.dump(validation_meta, f, indent=2)
    if SAVE_FINAL_MODEL:
        final_fit_and_save(train_df_reg, feature_cols, reg_template, all_algorithms, reg_feature_cols, validation_meta, problem_normalizer_params)
    print("\nDone.")


if __name__ == "__main__":
    main()
