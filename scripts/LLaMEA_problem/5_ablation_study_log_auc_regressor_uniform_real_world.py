
# fmt: off
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# fmt: on


MODEL_PATH = "data/Combined/models/bbob_mabbob_llm_mixed_auc_source_normalized_regressor_as_model.joblib"
ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"

OUT_DIR = "results_real_world_uniform_ela_ablation_log_auc"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_REPEATS = 10

# "all_real_world": uniform range from all real-world rows.
# "real_world": uniform range from the current real-world problem only.
UNIFORM_RANGE_SOURCE = "all_real_world"

FALLBACK_LOW = -1.0
FALLBACK_HIGH = 1.0
SAVE_LONG_PREDICTIONS = False
COMPUTE_SELECTOR_REGRET = True
AUC_MIN_POSITIVE = 1e-300


def harmonize_feature_names(df):
    rename = {}
    for c in df.columns:
        nc = c
        nc = nc.replace("ela_distribution.", "ela_distr.")
        nc = nc.replace("dispersion.", "disp.")
        nc = nc.replace("information_content.", "ic.")
        rename[c] = nc
    return df.rename(columns=rename)


def get_feature_group(feature_name):
    if "ela_meta" in feature_name:
        return "Meta-model"
    if "ela_distr" in feature_name or "ela_distribution" in feature_name:
        return "Distribution"
    if "ela_level" in feature_name:
        return "Level-set"
    if "nbc" in feature_name:
        return "Nearest Better"
    if "ic" in feature_name or "information_content" in feature_name:
        return "Info. Content"
    if "disp" in feature_name or "dispersion" in feature_name:
        return "Dispersion"
    if "pca" in feature_name:
        return "PCA"
    if "limo" in feature_name:
        return "Linear Model"
    if "cm_" in feature_name:
        return "Cell Mapping"
    if "gradient" in feature_name:
        return "Gradient"
    if "hill_climbing" in feature_name:
        return "Hill Climbing"
    if "length_scale" in feature_name:
        return "Length Scale"
    if "fla_metrics" in feature_name or "sobol" in feature_name:
        return "Sobol / FLA"
    return "Others"


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


def log_auc(auc):
    return np.log1p(np.maximum(np.asarray(auc, dtype=float), AUC_MIN_POSITIVE))


def inverse_log_auc(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, -745, 700)
    return np.expm1(y)


def safe_name(x):
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")


def ensure_real_world_ela_keys(ela_df):
    ela_df = harmonize_feature_names(ela_df.copy())
    ela_df = ela_df[pd.to_numeric(ela_df["fid"], errors="coerce") < 1].copy()

    ela_df["fid"] = pd.to_numeric(ela_df["fid"], errors="coerce").astype(int)
    ela_df["iid"] = pd.to_numeric(ela_df["iid"], errors="coerce").astype(int)
    ela_df["dim"] = pd.to_numeric(ela_df["dim"], errors="coerce").astype(int)

    if "problem_name" not in ela_df.columns:
        ela_df["problem_name"] = ela_df["fid"].apply(lambda x: f"REAL_{int(x)}")
    ela_df["problem_name"] = ela_df["problem_name"].fillna(
        ela_df["fid"].apply(lambda x: f"REAL_{int(x)}")
    )

    return ela_df


def ensure_real_world_perf_keys(perf_df):
    perf_df = perf_df.copy()
    perf_df = perf_df[pd.to_numeric(perf_df["fid"], errors="coerce") < 1].copy()

    perf_df["fid"] = pd.to_numeric(perf_df["fid"], errors="coerce").astype(int)
    perf_df["iid"] = pd.to_numeric(perf_df["iid"], errors="coerce").astype(int)
    perf_df["dim"] = pd.to_numeric(perf_df["dim"], errors="coerce").astype(int)

    if "problem_name" not in perf_df.columns:
        perf_df["problem_name"] = perf_df["fid"].apply(lambda x: f"REAL_{int(x)}")
    perf_df["problem_name"] = perf_df["problem_name"].fillna(
        perf_df["fid"].apply(lambda x: f"REAL_{int(x)}")
    )

    if "auc_mean" not in perf_df.columns:
        raise ValueError("Real-world performance table must contain auc_mean.")

    perf_df["auc_mean"] = pd.to_numeric(perf_df["auc_mean"], errors="coerce")
    valid = np.isfinite(perf_df["auc_mean"]) & (perf_df["auc_mean"] > AUC_MIN_POSITIVE)
    n0 = len(perf_df)
    perf_df = perf_df.loc[valid].copy()
    if len(perf_df) < n0:
        print(f"[Clean] Dropped {n0 - len(perf_df)} invalid real-world performance rows.")

    perf_df["true_log_auc"] = log_auc(perf_df["auc_mean"].to_numpy())
    return perf_df


def load_inputs():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    if not os.path.exists(ELA_PATH):
        raise FileNotFoundError(ELA_PATH)
    if not os.path.exists(PERF_PATH):
        raise FileNotFoundError(PERF_PATH)

    bundle = joblib.load(MODEL_PATH)
    reg = bundle["regressor"]
    feature_cols = bundle["feature_cols"]
    reg_feature_cols = bundle["reg_feature_cols"]
    algorithms = bundle["algorithms"]

    target_transform = bundle.get("target_transform", "log1p")
    if target_transform != "log1p":
        print(f"[Warn] Model target_transform={target_transform}; this script assumes log1p target.")

    ela_df = ensure_real_world_ela_keys(pd.read_csv(ELA_PATH))
    perf_df = ensure_real_world_perf_keys(pd.read_csv(PERF_PATH))

    real_df = pd.merge(
        perf_df,
        ela_df,
        on=["fid", "iid", "dim"],
        how="inner",
        suffixes=("_perf", ""),
    )
    print(ela_df)

    if "problem_name" not in real_df.columns:
        if "problem_name_perf" in real_df.columns:
            real_df["problem_name"] = real_df["problem_name_perf"]
        else:
            real_df["problem_name"] = real_df["fid"].apply(lambda x: f"REAL_{int(x)}")

    real_df["algname"] = real_df["algname"].astype(str)

    n0 = len(real_df)
    real_df = real_df[real_df["algname"].isin(algorithms)].copy()
    if len(real_df) < n0:
        print(f"[Clean] Dropped {n0 - len(real_df)} rows with algorithms unseen by the trained model.")

    if real_df.empty:
        raise RuntimeError("No matched real-world ELA/performance rows.")

    return reg, feature_cols, reg_feature_cols, algorithms, ela_df, perf_df, real_df


def prepare_X(df, feature_cols, reg_feature_cols, algorithms):
    available = [f for f in feature_cols if f in df.columns]
    missing = sorted(set(feature_cols) - set(available))

    X_ela = df[available].copy()
    for f in missing:
        X_ela[f] = 0.0
    X_ela = X_ela[feature_cols]
    X_ela = clean_X(X_ela)

    alg_dummies = pd.get_dummies(df["algname"].astype(str), prefix="algname")
    X = pd.concat([X_ela.reset_index(drop=True), alg_dummies.reset_index(drop=True)], axis=1)

    for alg in algorithms:
        c = f"algname_{alg}"
        if c not in X.columns:
            X[c] = 0.0

    for c in reg_feature_cols:
        if c not in X.columns:
            X[c] = 0.0

    return X[reg_feature_cols]


def get_uniform_range(feature, scope_df, global_real_ela):
    if UNIFORM_RANGE_SOURCE == "real_world":
        s = pd.to_numeric(scope_df[feature], errors="coerce") if feature in scope_df.columns else pd.Series(dtype=float)
    elif UNIFORM_RANGE_SOURCE == "all_real_world":
        s = pd.to_numeric(global_real_ela[feature], errors="coerce") if feature in global_real_ela.columns else pd.Series(dtype=float)
    else:
        raise ValueError(f"Unknown UNIFORM_RANGE_SOURCE: {UNIFORM_RANGE_SOURCE}")

    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return FALLBACK_LOW, FALLBACK_HIGH

    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return FALLBACK_LOW, FALLBACK_HIGH

    return lo, hi


def prediction_error_metrics(df):
    err = df["pred_log_auc"].to_numpy() - df["true_log_auc"].to_numpy()
    abs_err = np.abs(err)
    return {
        "log_mae": float(np.mean(abs_err)),
        "log_rmse": float(np.sqrt(np.mean(err ** 2))),
        "log_bias": float(np.mean(err)),
    }


def selector_metrics(df):
    rows = []
    for key, g in df.groupby(["problem_name", "fid", "iid", "dim"]):
        g = g.copy()
        pred_row = g.loc[g["pred_log_auc"].idxmin()]
        oracle_row = g.loc[g["auc_mean"].idxmin()]

        pred_actual_auc = float(pred_row["auc_mean"])
        oracle_actual_auc = float(oracle_row["auc_mean"])
        pred_log = float(log_auc(pred_actual_auc))
        oracle_log = float(log_auc(oracle_actual_auc))

        rows.append({
            "problem_name": key[0],
            "fid": int(key[1]),
            "iid": int(key[2]),
            "dim": int(key[3]),
            "pred_selected_alg": pred_row["algname"],
            "oracle_best_alg": oracle_row["algname"],
            "selected_is_oracle": pred_row["algname"] == oracle_row["algname"],
            "pred_selected_actual_auc": pred_actual_auc,
            "oracle_actual_auc": oracle_actual_auc,
            "log_regret_vs_oracle": pred_log - oracle_log,
            "relative_regret_vs_oracle": pred_actual_auc / oracle_actual_auc - 1.0 if oracle_actual_auc > AUC_MIN_POSITIVE else np.nan,
            "n_algorithms_available": int(len(g)),
        })
    return pd.DataFrame(rows)


def summarize_selector_ablation(selector_df, problem_name):
    base = selector_df[selector_df["ablation_feature"] == "__BASELINE__"].copy()

    base_log_regret = float(base["log_regret_vs_oracle"].mean()) if not base.empty else np.nan
    base_oracle_match = float(base["selected_is_oracle"].mean()) if not base.empty else np.nan

    rows = []
    for feat, sub in selector_df[selector_df["ablation_feature"] != "__BASELINE__"].groupby("ablation_feature"):
        rows.append({
            "problem_name": problem_name,
            "feature": feat,
            "feature_group": sub["feature_group"].iloc[0],
            "baseline_log_regret": base_log_regret,
            "ablated_log_regret_mean": float(sub["log_regret_vs_oracle"].mean()),
            "delta_log_regret_mean": float(sub["log_regret_vs_oracle"].mean() - base_log_regret),
            "baseline_oracle_match": base_oracle_match,
            "ablated_oracle_match_mean": float(sub["selected_is_oracle"].mean()),
            "delta_oracle_match_mean": float(sub["selected_is_oracle"].mean() - base_oracle_match),
            "n_measurements": int(len(sub)),
        })
    return pd.DataFrame(rows).sort_values("delta_log_regret_mean", ascending=False)


def plot_top_features(problem_name, result_df, top_k=20):
    df = result_df[result_df["feature"] != "__BASELINE__"].copy()
    top = df.sort_values("delta_log_mae_mean", ascending=False).head(top_k)
    top = top.sort_values("delta_log_mae_mean", ascending=True)

    plt.figure(figsize=(9, max(5, 0.35 * len(top))))
    plt.barh(top["feature"], top["delta_log_mae_mean"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Increase in log-AUC MAE after uniform feature perturbation")
    plt.ylabel("ELA feature")
    plt.title(f"Top {top_k} ELA features by error increase | {problem_name}")
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{safe_name(problem_name)}_uniform_top_features_log_auc_error.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")


def plot_groups(problem_name, group_df):
    if group_df.empty:
        return

    g = group_df.sort_values("delta_log_mae_sum", ascending=True)

    plt.figure(figsize=(8, max(4, 0.45 * len(g))))
    plt.barh(g["feature_group"], g["delta_log_mae_sum"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Sum of log-AUC MAE increase")
    plt.ylabel("ELA feature group")
    plt.title(f"ELA group importance by uniform perturbation | {problem_name}")
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{safe_name(problem_name)}_uniform_groups_log_auc_error.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")


def run_ablation_for_problem(problem_name, problem_df, reg, feature_cols, reg_feature_cols, algorithms, global_real_ela):
    rng = np.random.default_rng(RANDOM_SEED)
    problem_tag = safe_name(problem_name)
    print(f"\n[Problem] {problem_name} | rows={len(problem_df)}")

    X_base = prepare_X(problem_df, feature_cols, reg_feature_cols, algorithms)
    baseline = problem_df.copy()
    baseline["pred_log_auc"] = reg.predict(X_base)
    baseline["pred_raw_auc"] = inverse_log_auc(baseline["pred_log_auc"].to_numpy())
    baseline["ablation_feature"] = "__BASELINE__"
    baseline["feature_group"] = "__BASELINE__"
    baseline["repeat"] = -1

    base_err = prediction_error_metrics(baseline)

    rows = [{
        "problem_name": problem_name,
        "feature": "__BASELINE__",
        "feature_group": "__BASELINE__",
        "uniform_low": np.nan,
        "uniform_high": np.nan,
        "n_repeats": 1,
        "baseline_log_mae": base_err["log_mae"],
        "ablated_log_mae_mean": base_err["log_mae"],
        "delta_log_mae_mean": 0.0,
        "baseline_log_rmse": base_err["log_rmse"],
        "ablated_log_rmse_mean": base_err["log_rmse"],
        "delta_log_rmse_mean": 0.0,
        "baseline_log_bias": base_err["log_bias"],
        "ablated_log_bias_mean": base_err["log_bias"],
        "delta_log_bias_mean": 0.0,
    }]

    selector_rows = []
    if COMPUTE_SELECTOR_REGRET:
        base_selector = selector_metrics(baseline)
        base_selector["problem_name_group"] = problem_name
        base_selector["ablation_feature"] = "__BASELINE__"
        base_selector["feature_group"] = "__BASELINE__"
        base_selector["repeat"] = -1
        selector_rows.append(base_selector)

    long_parts = [baseline] if SAVE_LONG_PREDICTIONS else []

    for feat_i, feat in enumerate(feature_cols, start=1):
        if feat_i % 10 == 0 or feat_i == 1:
            print(f"  Feature {feat_i}/{len(feature_cols)}: {feat}")

        low, high = get_uniform_range(feat, problem_df, global_real_ela)
        repeat_metrics = []
        repeat_selector = []

        for rep in range(N_REPEATS):
            perturbed = problem_df.copy()
            perturbed[feat] = rng.uniform(low, high, size=len(perturbed))

            X_perm = prepare_X(perturbed, feature_cols, reg_feature_cols, algorithms)
            pred = perturbed.copy()
            pred["pred_log_auc"] = reg.predict(X_perm)
            pred["pred_raw_auc"] = inverse_log_auc(pred["pred_log_auc"].to_numpy())
            pred["ablation_feature"] = feat
            pred["feature_group"] = get_feature_group(feat)
            pred["repeat"] = rep

            repeat_metrics.append(prediction_error_metrics(pred))

            if COMPUTE_SELECTOR_REGRET:
                sel = selector_metrics(pred)
                sel["problem_name_group"] = problem_name
                sel["ablation_feature"] = feat
                sel["feature_group"] = get_feature_group(feat)
                sel["repeat"] = rep
                repeat_selector.append(sel)

            if SAVE_LONG_PREDICTIONS:
                long_parts.append(pred)

        met_df = pd.DataFrame(repeat_metrics)

        rows.append({
            "problem_name": problem_name,
            "feature": feat,
            "feature_group": get_feature_group(feat),
            "uniform_low": low,
            "uniform_high": high,
            "n_repeats": N_REPEATS,
            "baseline_log_mae": base_err["log_mae"],
            "ablated_log_mae_mean": float(met_df["log_mae"].mean()),
            "ablated_log_mae_std": float(met_df["log_mae"].std(ddof=0)),
            "delta_log_mae_mean": float(met_df["log_mae"].mean() - base_err["log_mae"]),
            "baseline_log_rmse": base_err["log_rmse"],
            "ablated_log_rmse_mean": float(met_df["log_rmse"].mean()),
            "ablated_log_rmse_std": float(met_df["log_rmse"].std(ddof=0)),
            "delta_log_rmse_mean": float(met_df["log_rmse"].mean() - base_err["log_rmse"]),
            "baseline_log_bias": base_err["log_bias"],
            "ablated_log_bias_mean": float(met_df["log_bias"].mean()),
            "ablated_log_bias_std": float(met_df["log_bias"].std(ddof=0)),
            "delta_log_bias_mean": float(met_df["log_bias"].mean() - base_err["log_bias"]),
        })

        if COMPUTE_SELECTOR_REGRET and repeat_selector:
            selector_rows.append(pd.concat(repeat_selector, ignore_index=True))

    result_df = pd.DataFrame(rows).sort_values("delta_log_mae_mean", ascending=False)
    result_path = os.path.join(OUT_DIR, f"{problem_tag}_uniform_feature_ablation_log_auc_error.csv")
    result_df.to_csv(result_path, index=False)
    print(f"Saved: {result_path}")

    group_df = (
        result_df[result_df["feature"] != "__BASELINE__"]
        .groupby("feature_group", as_index=False)
        .agg(
            delta_log_mae_mean=("delta_log_mae_mean", "mean"),
            delta_log_mae_sum=("delta_log_mae_mean", "sum"),
            delta_log_rmse_mean=("delta_log_rmse_mean", "mean"),
            n_features=("feature", "count"),
        )
        .sort_values("delta_log_mae_sum", ascending=False)
    )
    group_path = os.path.join(OUT_DIR, f"{problem_tag}_uniform_group_ablation_log_auc_error.csv")
    group_df.to_csv(group_path, index=False)
    print(f"Saved: {group_path}")

    selector_df = None
    if COMPUTE_SELECTOR_REGRET and selector_rows:
        selector_df = pd.concat(selector_rows, ignore_index=True)
        selector_path = os.path.join(OUT_DIR, f"{problem_tag}_uniform_selector_ablation.csv")
        selector_df.to_csv(selector_path, index=False)
        print(f"Saved: {selector_path}")

        selector_summary = summarize_selector_ablation(selector_df, problem_name)
        selector_summary_path = os.path.join(OUT_DIR, f"{problem_tag}_uniform_selector_ablation_summary.csv")
        selector_summary.to_csv(selector_summary_path, index=False)
        print(f"Saved: {selector_summary_path}")

    if SAVE_LONG_PREDICTIONS and long_parts:
        long_df = pd.concat(long_parts, ignore_index=True)
        long_path = os.path.join(OUT_DIR, f"{problem_tag}_uniform_ablation_long_predictions.csv")
        long_df.to_csv(long_path, index=False)
        print(f"Saved: {long_path}")

    plot_top_features(problem_name, result_df)
    plot_groups(problem_name, group_df)

    return result_df, group_df, selector_df


def plot_overall_group_summary(overall_group):
    if overall_group.empty:
        return

    g = overall_group.sort_values("delta_log_mae_sum_mean", ascending=True)

    plt.figure(figsize=(8, max(4, 0.45 * len(g))))
    plt.barh(g["feature_group"], g["delta_log_mae_sum_mean"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Mean group-level log-AUC MAE increase across real-world problems")
    plt.ylabel("ELA feature group")
    plt.title("Overall real-world ELA group ablation")
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "overall_uniform_group_ablation_log_auc_error.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")


def main():
    reg, feature_cols, reg_feature_cols, algorithms, real_ela, real_perf, real_df = load_inputs()

    print("\n=== Real-world uniform ELA ablation ===")
    print(f"Matched rows: {len(real_df)}")
    print(f"Real-world problems: {real_df['problem_name'].nunique()}")
    print(f"Algorithms in model: {len(algorithms)}")
    print(f"Features: {len(feature_cols)}")
    print(f"N_REPEATS: {N_REPEATS}")
    print(f"UNIFORM_RANGE_SOURCE: {UNIFORM_RANGE_SOURCE}")

    all_feature_results = []
    all_group_results = []
    all_selector_results = []

    for problem_name, problem_df in real_df.groupby("problem_name"):
        feature_df, group_df, selector_df = run_ablation_for_problem(
            problem_name=problem_name,
            problem_df=problem_df,
            reg=reg,
            feature_cols=feature_cols,
            reg_feature_cols=reg_feature_cols,
            algorithms=algorithms,
            global_real_ela=real_ela,
        )

        all_feature_results.append(feature_df)
        all_group_results.append(group_df.assign(problem_name=problem_name))

        if selector_df is not None:
            all_selector_results.append(selector_df.assign(problem_name_group=problem_name))

    if all_feature_results:
        all_feature_df = pd.concat(all_feature_results, ignore_index=True)
        out = os.path.join(OUT_DIR, "all_real_world_uniform_feature_ablation_log_auc_error.csv")
        all_feature_df.to_csv(out, index=False)
        print(f"Saved: {out}")

    if all_group_results:
        all_group_df = pd.concat(all_group_results, ignore_index=True)
        out = os.path.join(OUT_DIR, "all_real_world_uniform_group_ablation_log_auc_error.csv")
        all_group_df.to_csv(out, index=False)
        print(f"Saved: {out}")

        overall_group = (
            all_group_df
            .groupby("feature_group", as_index=False)
            .agg(
                delta_log_mae_mean=("delta_log_mae_mean", "mean"),
                delta_log_mae_sum_mean=("delta_log_mae_sum", "mean"),
                delta_log_rmse_mean=("delta_log_rmse_mean", "mean"),
                n_features_mean=("n_features", "mean"),
            )
            .sort_values("delta_log_mae_sum_mean", ascending=False)
        )
        out = os.path.join(OUT_DIR, "overall_uniform_group_ablation_log_auc_error.csv")
        overall_group.to_csv(out, index=False)
        print(f"Saved: {out}")
        plot_overall_group_summary(overall_group)

    if all_selector_results:
        all_selector_df = pd.concat(all_selector_results, ignore_index=True)
        out = os.path.join(OUT_DIR, "all_real_world_uniform_selector_ablation.csv")
        all_selector_df.to_csv(out, index=False)
        print(f"Saved: {out}")

        summaries = []
        for problem_name, sub in all_selector_df.groupby("problem_name_group"):
            summaries.append(summarize_selector_ablation(sub, problem_name))
        summary_df = pd.concat(summaries, ignore_index=True)
        out = os.path.join(OUT_DIR, "all_real_world_uniform_selector_ablation_summary.csv")
        summary_df.to_csv(out, index=False)
        print(f"Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
