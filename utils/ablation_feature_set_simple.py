
# fmt: off
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# fmt: on


# ============================================================
# 1. Paths
# ============================================================

MODEL_PATH = "data/Combined/models/bbob_mabbob_llm_mixed_auc_per_problem_normalized_regressor_as_model.joblib"

ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"

OUT_DIR = "results/Combined/feature_set_ablation_simple"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. Config
# ============================================================

RANDOM_SEED = 42
N_REPEATS = 5

# "uniform": replace features by uniform random values sampled from real-world min/max.
# "permutation": shuffle feature values inside the current real-world problem.
ABLATION_MODE = "uniform"

# Optional CSV with columns:
#   feature_set, feature
# If None, the script uses DEFAULT_FEATURE_SETS.
FEATURE_SET_CSV = None

# Fill this if you want custom feature sets.
# Example:
# DEFAULT_FEATURE_SETS = {
#     "ABLATION_TOP_20": ["ela_meta.lin_simple.adj_r2", "pca.expl_var_PC1"],
#     "BOTTOM_20": ["nbc.nn_nb.sd_ratio", "disp.ratio_mean_02"],
# }
DEFAULT_FEATURE_SETS = {}

# If no DEFAULT_FEATURE_SETS and no FEATURE_SET_CSV are given,
# automatically use ELA feature groups as feature sets.
AUTO_GROUP_FEATURE_SETS = True

# Always include all features as one set.
INCLUDE_FULL_FEATURE_SET = True


# ============================================================
# 3. Load model
# ============================================================

bundle = joblib.load(MODEL_PATH)
reg = bundle["regressor"]
feature_cols = bundle["feature_cols"]
reg_feature_cols = bundle["reg_feature_cols"]

# Prefer algorithm list from bundle, otherwise infer from reg_feature_cols.
if "algorithms" in bundle:
    algnames = list(bundle["algorithms"])
else:
    algnames = [c.replace("algname_", "") for c in reg_feature_cols if c.startswith("algname_")]


def force_single_thread_model(model):
    if hasattr(model, "get_params") and hasattr(model, "set_params"):
        params = model.get_params()
        if "n_jobs" in params:
            model.set_params(n_jobs=1)
        nested_n_jobs = {key: 1 for key in params if key.endswith("__n_jobs")}
        if nested_n_jobs:
            model.set_params(**nested_n_jobs)
    return model


reg = force_single_thread_model(reg)


# ============================================================
# 4. ELA helpers
# ============================================================

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
    return "Others"


def clean_X_base(X):
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.astype(float)


def prepare_base_ela_matrix(prob_data):
    available_features = [f for f in feature_cols if f in prob_data.columns]
    missing_features = sorted(set(feature_cols) - set(available_features))

    if missing_features:
        print(f"Warning: {len(missing_features)} features missing, will be filled as 0.")

    X_base = prob_data[available_features].copy()
    X_base = clean_X_base(X_base)

    for f in missing_features:
        X_base[f] = 0.0

    X_base = X_base[feature_cols]
    return X_base


def make_model_input(X_ela, alg):
    X_input = X_ela.copy()

    # Add algorithm one-hot columns.
    for a in algnames:
        X_input[f"algname_{a}"] = 1.0 if a == alg else 0.0

    # Add any missing trained columns.
    for c in reg_feature_cols:
        if c not in X_input.columns:
            X_input[c] = 0.0

    return X_input[reg_feature_cols]


def predict_all_algorithms_mean(X_ela):
    preds = []
    for alg in algnames:
        X_input = make_model_input(X_ela, alg)
        pred = reg.predict(X_input)
        preds.append({
            "algname": alg,
            "pred_mean": float(np.mean(pred)),
            "pred_median": float(np.median(pred)),
        })
    return pd.DataFrame(preds)


def get_uniform_ranges(all_real_ela):
    ranges = {}
    for f in feature_cols:
        if f not in all_real_ela.columns:
            ranges[f] = (-1.0, 1.0)
            continue
        s = pd.to_numeric(all_real_ela[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            ranges[f] = (-1.0, 1.0)
            continue
        lo, hi = float(s.min()), float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            ranges[f] = (-1.0, 1.0)
        else:
            ranges[f] = (lo, hi)
    return ranges


# ============================================================
# 5. Feature sets
# ============================================================

def load_feature_sets():
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

    # Remove empty feature sets.
    feature_sets = {name: feats for name, feats in feature_sets.items() if len(feats) > 0}

    return feature_sets


# ============================================================
# 6. Ablation
# ============================================================

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


def run_feature_set_ablation(problem_name, prob_data, feature_sets, uniform_ranges):
    print(f"Running feature-set ablation on [{problem_name}]...")

    if prob_data.empty:
        return None

    rng = np.random.default_rng(RANDOM_SEED)

    X_base = prepare_base_ela_matrix(prob_data)

    base_pred_df = predict_all_algorithms_mean(X_base)
    base_overall_mean = float(base_pred_df["pred_mean"].mean())
    base_best_row = base_pred_df.loc[base_pred_df["pred_mean"].idxmin()]
    base_best_alg = base_best_row["algname"]
    base_best_pred = float(base_best_row["pred_mean"])

    base_pred_df.to_csv(
        os.path.join(OUT_DIR, f"{problem_name}_baseline_algorithm_predictions.csv"),
        index=False,
    )

    rows = []

    for set_name, feats in feature_sets.items():
        impacts_mean_all_algs = []
        impacts_best_alg = []
        changed_best_alg = []

        for rep in range(N_REPEATS):
            X_ab = ablate_feature_set(X_base, feats, rng, uniform_ranges)
            ab_pred_df = predict_all_algorithms_mean(X_ab)

            ab_overall_mean = float(ab_pred_df["pred_mean"].mean())
            ab_best_row = ab_pred_df.loc[ab_pred_df["pred_mean"].idxmin()]
            ab_best_alg = ab_best_row["algname"]
            ab_best_pred = float(ab_best_row["pred_mean"])

            impacts_mean_all_algs.append(ab_overall_mean - base_overall_mean)
            impacts_best_alg.append(ab_best_pred - base_best_pred)
            changed_best_alg.append(ab_best_alg != base_best_alg)

        rows.append({
            "problem_name": problem_name,
            "feature_set": set_name,
            "n_features": len(feats),
            "feature_groups": ";".join(sorted(set(get_feature_group(f) for f in feats))),
            "baseline_mean_pred_all_algs": base_overall_mean,
            "ablated_mean_pred_all_algs_mean": float(np.mean([base_overall_mean + x for x in impacts_mean_all_algs])),
            "impact_mean_pred_all_algs_mean": float(np.mean(impacts_mean_all_algs)),
            "impact_mean_pred_all_algs_std": float(np.std(impacts_mean_all_algs)),
            "baseline_best_alg": base_best_alg,
            "baseline_best_pred": base_best_pred,
            "impact_best_pred_mean": float(np.mean(impacts_best_alg)),
            "impact_best_pred_std": float(np.std(impacts_best_alg)),
            "best_alg_change_rate": float(np.mean(changed_best_alg)),
            "n_repeats": N_REPEATS,
            "ablation_mode": ABLATION_MODE,
        })

    res = pd.DataFrame(rows)
    res = res.sort_values("impact_mean_pred_all_algs_mean", ascending=False)

    out_csv = os.path.join(OUT_DIR, f"{problem_name}_feature_set_ablation.csv")
    res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot impact on mean predicted target over all algorithms.
    plot_df = res.sort_values("impact_mean_pred_all_algs_mean", ascending=True)
    plt.figure(figsize=(10, max(5, 0.45 * len(plot_df))))
    plt.barh(plot_df["feature_set"], plot_df["impact_mean_pred_all_algs_mean"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Change in mean predicted target after feature-set ablation")
    plt.ylabel("Feature set")
    plt.title(f"Feature-set ablation impact | {problem_name}")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{problem_name}_feature_set_ablation_impact.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")

    # Plot best algorithm change rate.
    plot_df = res.sort_values("best_alg_change_rate", ascending=True)
    plt.figure(figsize=(10, max(5, 0.45 * len(plot_df))))
    plt.barh(plot_df["feature_set"], plot_df["best_alg_change_rate"])
    plt.xlabel("Rate that predicted best algorithm changes")
    plt.ylabel("Feature set")
    plt.title(f"Feature-set ablation changes selected algorithm | {problem_name}")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{problem_name}_feature_set_best_alg_change_rate.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")

    return res


# ============================================================
# 7. Main
# ============================================================

def main():
    ela_df = pd.read_csv(ELA_PATH)
    ela_df = harmonize_feature_names(ela_df)

    if "fid" not in ela_df.columns:
        raise ValueError("ELA table must contain fid.")

    my_problems = ela_df[pd.to_numeric(ela_df["fid"], errors="coerce") < 1].copy()

    if "problem_name" not in my_problems.columns:
        my_problems["problem_name"] = my_problems["fid"].apply(lambda x: f"REAL_{int(x)}")

    if my_problems.empty:
        raise RuntimeError("No real-world problems found with fid < 1 in ELA table.")

    feature_sets = load_feature_sets()
    if not feature_sets:
        raise RuntimeError("No valid feature sets found.")

    pd.DataFrame(
        [{"feature_set": name, "feature": f} for name, feats in feature_sets.items() for f in feats]
    ).to_csv(os.path.join(OUT_DIR, "feature_sets_used.csv"), index=False)

    uniform_ranges = get_uniform_ranges(my_problems)

    print("=== Simple feature-set ablation ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Real-world problems: {my_problems['problem_name'].nunique()}")
    print(f"Feature sets: {len(feature_sets)}")
    print(f"Algorithms: {len(algnames)}")
    print(f"Ablation mode: {ABLATION_MODE}")
    print(f"N_REPEATS: {N_REPEATS}")

    all_results = []

    for problem_name in my_problems["problem_name"].dropna().unique():
        prob_data = my_problems[my_problems["problem_name"] == problem_name].copy()
        res = run_feature_set_ablation(problem_name, prob_data, feature_sets, uniform_ranges)
        if res is not None:
            all_results.append(res)

    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        out = os.path.join(OUT_DIR, "all_real_world_feature_set_ablation.csv")
        all_df.to_csv(out, index=False)
        print(f"Saved: {out}")

        overall = (
            all_df.groupby("feature_set", as_index=False)
            .agg(
                n_features=("n_features", "first"),
                impact_mean_pred_all_algs_mean=("impact_mean_pred_all_algs_mean", "mean"),
                impact_mean_pred_all_algs_std=("impact_mean_pred_all_algs_mean", "std"),
                impact_best_pred_mean=("impact_best_pred_mean", "mean"),
                impact_best_pred_std=("impact_best_pred_mean", "std"),
                best_alg_change_rate_mean=("best_alg_change_rate", "mean"),
                n_problems=("problem_name", "nunique"),
            )
            .sort_values("impact_mean_pred_all_algs_mean", ascending=False)
        )
        out = os.path.join(OUT_DIR, "overall_feature_set_ablation.csv")
        overall.to_csv(out, index=False)
        print(f"Saved: {out}")

        plot_df = overall.sort_values("impact_mean_pred_all_algs_mean", ascending=True)
        plt.figure(figsize=(10, max(5, 0.45 * len(plot_df))))
        plt.barh(plot_df["feature_set"], plot_df["impact_mean_pred_all_algs_mean"])
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Mean change in predicted target across real-world problems")
        plt.ylabel("Feature set")
        plt.title("Overall feature-set ablation impact")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "overall_feature_set_ablation_impact.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved: {out_png}")

        plot_df = overall.sort_values("best_alg_change_rate_mean", ascending=True)
        plt.figure(figsize=(10, max(5, 0.45 * len(plot_df))))
        plt.barh(plot_df["feature_set"], plot_df["best_alg_change_rate_mean"])
        plt.xlabel("Mean best-algorithm change rate across real-world problems")
        plt.ylabel("Feature set")
        plt.title("Overall feature-set ablation effect on selected algorithm")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "overall_feature_set_best_alg_change_rate.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved: {out_png}")

    print("Done.")


if __name__ == "__main__":
    main()
