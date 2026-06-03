
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


# ============================================================
# 1. Paths
# ============================================================

BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
BBOB_PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"

MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
MABBOB_PERF_PATH = "data/MABBOB/mabbob_algorithm_auc_performance.csv"

LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"
LLM_PERF_PATH = "data/LLM/llm_algorithm_auc_performance.csv"

OUT_DIR = "results/ela_space_compare/"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Outlier filtering for visualization
# ============================================================
# Keep the full PCA coordinate table, but additionally create filtered plots
# after removing visually extreme points. By default only LLM extremes are
# filtered because those are the points that often dominate the axis limits.
FILTER_OUTLIERS_FOR_PLOTS = True
FILTER_SOURCES = ["LLM"]      # change to ["BBOB", "MABBOB", "LLM"] for all sources
OUTLIER_METHOD = "robust_distance"  # "robust_distance" or "pc_quantile"
ROBUST_DISTANCE_Q = 0.95     # remove top 2.5% largest robust-distance points per source
PC_QUANTILE_LOW = 0.01        # used only when OUTLIER_METHOD == "pc_quantile"
PC_QUANTILE_HIGH = 0.99       # used only when OUTLIER_METHOD == "pc_quantile"


# ============================================================
# 2. Helpers
# ============================================================

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


def drop_invalid_ela_rows(df, problem_type):
    df = df.copy()
    n0 = len(df)

    if "FAILED" in df.columns:
        failed_mask = pd.to_numeric(df["FAILED"], errors="coerce").fillna(0) != 0
        df = df.loc[~failed_mask].copy()

    if "dim" not in df.columns:
        raise ValueError(f"{problem_type} ELA table has no dim column.")

    dim_num = pd.to_numeric(df["dim"], errors="coerce")
    valid_dim = np.isfinite(dim_num) & (dim_num > 0)
    df = df.loc[valid_dim].copy()
    df["dim"] = dim_num.loc[df.index].astype(int)

    n_drop = n0 - len(df)
    if n_drop > 0:
        print(f"[Clean] Dropped {n_drop} invalid {problem_type} ELA rows; kept {len(df)} / {n0}.")

    return df


def ensure_problem_keys(df, problem_type):
    df = df.copy()
    df = harmonize_feature_names(df)
    df = drop_invalid_ela_rows(df, problem_type)

    df["problem_type"] = problem_type

    if problem_type == "BBOB":
        df = df[df["fid"].between(1, 24)].copy()
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype(int)
        if "problem_name" not in df.columns:
            df["problem_name"] = df["fid"].apply(lambda x: f"BBOB_F{int(x)}")
        df["problem_name"] = df["problem_name"].fillna(
            df["fid"].apply(lambda x: f"BBOB_F{int(x)}")
        )

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
        df = df.loc[np.isfinite(id_num)].copy()
        df["llm_problem_id"] = id_num.loc[df.index].astype(int)

        df["fid"] = -200
        df["iid"] = df["llm_problem_id"].astype(int)
        if "problem_name" not in df.columns:
            df["problem_name"] = df["iid"].apply(lambda x: f"LLM_{int(x)}")
        df["problem_name"] = df["problem_name"].fillna(
            df["iid"].apply(lambda x: f"LLM_{int(x)}")
        )

    else:
        raise ValueError(problem_type)

    df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype(int)
    df["source_dataset"] = problem_type
    return df


def ensure_perf_keys(df, problem_type):
    df = df.copy()
    df["problem_type"] = problem_type

    if problem_type == "BBOB":
        df = df[df["fid"].between(1, 24)].copy()
        df["fid"] = pd.to_numeric(df["fid"], errors="coerce").astype(int)
        df["iid"] = pd.to_numeric(df["iid"], errors="coerce").astype(int)

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

    elif problem_type == "LLM":
        if "llm_problem_id" not in df.columns:
            if "iid" in df.columns:
                df["llm_problem_id"] = df["iid"]
            else:
                raise ValueError("LLM performance must contain llm_problem_id or iid.")

        id_num = pd.to_numeric(df["llm_problem_id"], errors="coerce")
        df = df.loc[np.isfinite(id_num)].copy()
        df["llm_problem_id"] = id_num.loc[df.index].astype(int)

        df["fid"] = -200
        df["iid"] = df["llm_problem_id"].astype(int)

    else:
        raise ValueError(problem_type)

    df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype(int)

    if "auc_mean" in df.columns:
        auc_num = pd.to_numeric(df["auc_mean"], errors="coerce")
        df = df.loc[np.isfinite(auc_num)].copy()
        df["auc_mean"] = auc_num.loc[df.index].astype(float)

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


def load_matched_ela_rows():
    for p in [
        BBOB_ELA_PATH,
        BBOB_PERF_PATH,
        MABBOB_ELA_PATH,
        MABBOB_PERF_PATH,
        LLM_ELA_PATH,
        LLM_PERF_PATH,
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

    perf_keys = perf_df[["problem_type", "fid", "iid", "dim"]].drop_duplicates()

    matched_ela = pd.merge(
        ela_df,
        perf_keys,
        on=["problem_type", "fid", "iid", "dim"],
        how="inner",
    )

    matched_ela = matched_ela.drop_duplicates(
        subset=["problem_type", "fid", "iid", "dim", "problem_name"]
    ).reset_index(drop=True)

    return matched_ela, perf_df


# ============================================================
# 3. Robust ELA matrix for PCA
# ============================================================

def robust_log_signed_transform(X):
    """
    Make extreme ELA values PCA-safe.

    ELA features may contain finite but astronomically large values.
    StandardScaler can overflow while computing variance. We therefore:
      - coerce to numeric;
      - replace inf by NaN;
      - robustly clip each column;
      - apply signed log1p;
      - impute;
      - robust-scale.
    """
    X = X.copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    dropped_cols = []
    for c in list(X.columns):
        s = X[c]
        finite = s[np.isfinite(s)]

        if len(finite) == 0 or finite.nunique(dropna=True) <= 1:
            dropped_cols.append(c)
            X = X.drop(columns=[c])
            continue

        lo = finite.quantile(0.001)
        hi = finite.quantile(0.999)

        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = finite.min()
            hi = finite.max()

        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            X[c] = s.clip(lower=lo, upper=hi)

    # Absolute hard clip before log transform.
    X = X.clip(lower=-1e100, upper=1e100)

    # Signed log transform preserves sign and compresses magnitude.
    X = np.sign(X) * np.log1p(np.abs(X))

    X = pd.DataFrame(X, columns=[c for c in X.columns], index=X.index)
    X = X.replace([np.inf, -np.inf], np.nan)

    if dropped_cols:
        print(f"[Clean] Dropped {len(dropped_cols)} empty/constant/pathological feature cols before PCA.")

    return X


def build_feature_matrix(matched_ela, feature_cols):
    X_raw = matched_ela[feature_cols].copy()

    X_trans = robust_log_signed_transform(X_raw)

    # Some columns may have been dropped by robust transform.
    used_feature_cols = X_trans.columns.tolist()

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_trans)

    # Guard after imputation.
    X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)

    # RobustScaler is safer than StandardScaler for very heavy-tailed ELA.
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
    X_scaled = scaler.fit_transform(X_imp)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Final mild clipping so PCA cannot receive pathological values.
    X_scaled = np.clip(X_scaled, -1e6, 1e6)

    if not np.all(np.isfinite(X_scaled)):
        raise RuntimeError("X_scaled still contains non-finite values after robust cleaning.")

    return X_scaled, used_feature_cols



def filter_pca_outliers_for_plot(df_plot):
    """
    Remove extreme points for visualization only.

    The full coordinate table is still saved separately. This function saves:
      - ela_space_outliers_removed_for_plot.csv
      - ela_space_pca_coordinates_filtered_for_plot.csv

    Default behavior removes only extreme LLM points, using robust distance
    in the PC1/PC2 space within each source.
    """
    if not FILTER_OUTLIERS_FOR_PLOTS:
        return df_plot.copy(), pd.DataFrame(columns=df_plot.columns.tolist() + ["outlier_score"])

    keep_mask = pd.Series(True, index=df_plot.index)
    outlier_parts = []

    for source in FILTER_SOURCES:
        idx = df_plot.index[df_plot["problem_type"] == source]
        if len(idx) < 10:
            continue

        sub = df_plot.loc[idx, ["PC1", "PC2"]].copy()

        if OUTLIER_METHOD == "pc_quantile":
            q1_low, q1_high = sub["PC1"].quantile([PC_QUANTILE_LOW, PC_QUANTILE_HIGH])
            q2_low, q2_high = sub["PC2"].quantile([PC_QUANTILE_LOW, PC_QUANTILE_HIGH])

            source_keep = (
                sub["PC1"].between(q1_low, q1_high)
                & sub["PC2"].between(q2_low, q2_high)
            )

            score = pd.Series(0.0, index=sub.index)
            score.loc[~source_keep] = 1.0

        elif OUTLIER_METHOD == "robust_distance":
            med = sub.median(axis=0)
            mad = (sub - med).abs().median(axis=0)
            mad = mad.replace(0, np.nan)

            # Fallback if one dimension has zero MAD.
            scale = mad.fillna(sub.std(axis=0)).replace(0, 1.0)
            z = (sub - med) / scale
            score = np.sqrt((z ** 2).sum(axis=1))

            threshold = score.quantile(ROBUST_DISTANCE_Q)
            source_keep = score <= threshold

        else:
            raise ValueError(f"Unknown OUTLIER_METHOD: {OUTLIER_METHOD}")

        drop_idx = source_keep.index[~source_keep]
        keep_mask.loc[drop_idx] = False

        if len(drop_idx) > 0:
            out_sub = df_plot.loc[drop_idx].copy()
            out_sub["outlier_score"] = score.loc[drop_idx].values
            outlier_parts.append(out_sub)

        print(
            f"[Outlier filter] {source}: removed {len(drop_idx)} / {len(idx)} "
            f"points using {OUTLIER_METHOD}."
        )

    filtered = df_plot.loc[keep_mask].copy().reset_index(drop=True)

    if outlier_parts:
        outliers = pd.concat(outlier_parts, ignore_index=True)
    else:
        outliers = pd.DataFrame(columns=df_plot.columns.tolist() + ["outlier_score"])

    filtered.to_csv(
        os.path.join(OUT_DIR, "ela_space_pca_coordinates_filtered_for_plot.csv"),
        index=False,
    )
    outliers.to_csv(
        os.path.join(OUT_DIR, "ela_space_outliers_removed_for_plot.csv"),
        index=False,
    )

    print(f"[Outlier filter] Kept {len(filtered)} / {len(df_plot)} points for filtered plots.")
    return filtered, outliers



# ============================================================
# 4. Plotting and metrics
# ============================================================

def covariance_ellipse(ax, points, n_std=2.0):
    if len(points) < 3:
        return

    cov = np.cov(points[:, 0], points[:, 1])
    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    if np.any(vals <= 0):
        return

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    mean = points.mean(axis=0)

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        fill=False,
        linewidth=1.5,
        alpha=0.8,
    )
    ax.add_patch(ell)


def plot_pca_scatter(df_plot, pca_model, suffix=""):
    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw BBOB last so it is not hidden by the usually larger LLM cloud.
    source_order = ["LLM", "MABBOB", "BBOB"]
    style = {
        "BBOB":   {"marker": "o", "s": 38, "alpha": 0.85, "linewidths": 0.35, "edgecolors": "black"},
        "MABBOB": {"marker": "^", "s": 30, "alpha": 0.65, "linewidths": 0.20, "edgecolors": "black"},
        "LLM":    {"marker": ".", "s": 22, "alpha": 0.38, "linewidths": 0.00, "edgecolors": "none"},
    }

    for source in source_order:
        sub = df_plot[df_plot["problem_type"] == source]
        kwargs = style[source]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            label=f"{source} (n={len(sub)})",
            **kwargs,
        )

        pts = sub[["PC1", "PC2"]].to_numpy()
        if len(pts) > 0:
            centroid = pts.mean(axis=0)
            # ax.scatter([centroid[0]], [centroid[1]], marker="x", s=140, linewidths=2.0)
            # covariance_ellipse(ax, pts, n_std=2.0)

    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title("BBOB vs MA-BBOB vs LLM problems in robust ELA-PCA space\n(only instances used by regressor)")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"ela_space_pca_scatter{suffix}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_source_panels(df_plot, pca_model, suffix=""):
    source_order = ["BBOB", "MABBOB", "LLM"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)

    x_all = df_plot["PC1"].to_numpy()
    y_all = df_plot["PC2"].to_numpy()
    x_margin = 0.05 * (x_all.max() - x_all.min() + 1e-12)
    y_margin = 0.05 * (y_all.max() - y_all.min() + 1e-12)

    for ax, source in zip(axes, source_order):
        sub = df_plot[df_plot["problem_type"] == source]

        if len(sub) > 0:
            ax.hexbin(sub["PC1"], sub["PC2"], gridsize=25, mincnt=1)
            pts = sub[["PC1", "PC2"]].to_numpy()
            centroid = pts.mean(axis=0)
            # ax.scatter([centroid[0]], [centroid[1]], marker="x", s=120)
            # covariance_ellipse(ax, pts, n_std=2.0)

        ax.set_title(f"{source} (n={len(sub)})")
        ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.grid(alpha=0.25)
        ax.set_xlim(x_all.min() - x_margin, x_all.max() + x_margin)
        ax.set_ylim(y_all.min() - y_margin, y_all.max() + y_margin)

    axes[0].set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"ela_space_pca_panels{suffix}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def compute_distance_summary(df_plot, X_scaled):
    sources = ["BBOB", "MABBOB", "LLM"]
    rows = []

    for src in sources:
        idx_src = df_plot.index[df_plot["problem_type"] == src].to_numpy()
        X_src = X_scaled[idx_src]

        for tgt in sources:
            if src == tgt:
                continue

            idx_tgt = df_plot.index[df_plot["problem_type"] == tgt].to_numpy()
            X_tgt = X_scaled[idx_tgt]

            if len(X_src) == 0 or len(X_tgt) == 0:
                continue

            D = pairwise_distances(X_src, X_tgt, metric="euclidean")
            nn = D.min(axis=1)

            rows.append({
                "source": src,
                "target": tgt,
                "source_n": int(len(X_src)),
                "target_n": int(len(X_tgt)),
                "nn_distance_mean": float(np.mean(nn)),
                "nn_distance_median": float(np.median(nn)),
                "nn_distance_std": float(np.std(nn)),
                "nn_distance_q10": float(np.quantile(nn, 0.10)),
                "nn_distance_q90": float(np.quantile(nn, 0.90)),
            })

    summary = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "pairwise_nn_distance_summary.csv")
    summary.to_csv(out, index=False)
    print(f"Saved: {out}")
    return summary


def save_basic_counts(df_plot, perf_df, feature_cols, used_feature_cols, pca_model):
    counts = (
        df_plot.groupby("problem_type")
        .agg(
            n_instances=("problem_name", "count"),
            n_unique_problem_names=("problem_name", "nunique"),
            dim_min=("dim", "min"),
            dim_max=("dim", "max"),
        )
        .reset_index()
    )

    counts["n_feature_cols_initial"] = len(feature_cols)
    counts["n_feature_cols_used_after_cleaning"] = len(used_feature_cols)
    counts["pca_var_pc1"] = float(pca_model.explained_variance_ratio_[0])
    counts["pca_var_pc2"] = float(pca_model.explained_variance_ratio_[1])

    out = os.path.join(OUT_DIR, "matched_instance_counts.csv")
    counts.to_csv(out, index=False)
    print(f"Saved: {out}")

    perf_counts = (
        perf_df.groupby("problem_type")
        .agg(
            perf_rows=("auc_mean", "count"),
            unique_problem_keys=("iid", "nunique"),
            unique_algorithms=("algname", "nunique"),
        )
        .reset_index()
    )
    out2 = os.path.join(OUT_DIR, "performance_counts.csv")
    perf_counts.to_csv(out2, index=False)
    print(f"Saved: {out2}")



def plot_bbob_highlight(df_plot, pca_model, suffix=""):
    """
    Plot all non-BBOB points in the background and BBOB on top.
    Useful when BBOB is present but visually hidden by a larger LLM cloud.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    bg = df_plot[df_plot["problem_type"] != "BBOB"]
    bbob = df_plot[df_plot["problem_type"] == "BBOB"]

    ax.scatter(
        bg["PC1"],
        bg["PC2"],
        s=16,
        alpha=0.20,
        marker=".",
        label=f"Non-BBOB background (n={len(bg)})",
    )

    ax.scatter(
        bbob["PC1"],
        bbob["PC2"],
        s=42,
        alpha=0.90,
        marker="o",
        edgecolors="black",
        linewidths=0.35,
        label=f"BBOB (n={len(bbob)})",
    )

    if len(bbob) > 0:
        pts = bbob[["PC1", "PC2"]].to_numpy()
        centroid = pts.mean(axis=0)
        # ax.scatter([centroid[0]], [centroid[1]], marker="x", s=160, linewidths=2.0)
        # covariance_ellipse(ax, pts, n_std=2.0)

    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title("BBOB highlighted in robust ELA-PCA space")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"ela_space_bbob_highlight{suffix}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_pairwise_overlays(df_plot, pca_model, suffix=""):
    """
    Three pairwise overlays make it easier to compare BBOB against each source.
    """
    pairs = [("BBOB", "MABBOB"), ("BBOB", "LLM"), ("MABBOB", "LLM")]

    for a, b in pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        for source, marker, alpha, size in [(a, "o", 0.75, 34), (b, "^", 0.55, 28)]:
            sub = df_plot[df_plot["problem_type"] == source]
            # ax.scatter(
            #     sub["PC1"],
            #     sub["PC2"],
            #     s=size,
            #     alpha=alpha,
            #     marker=marker,
            #     edgecolors="black",
            #     linewidths=0.25,
            #     label=f"{source} (n={len(sub)})",
            # )
            # pts = sub[["PC1", "PC2"]].to_numpy()
            # if len(pts) > 0:
            #     covariance_ellipse(ax, pts, n_std=2.0)

        ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% var)")
        ax.set_title(f"{a} vs {b} in robust ELA-PCA space")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()

        out = os.path.join(OUT_DIR, f"ela_space_{a.lower()}_vs_{b.lower()}{suffix}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved: {out}")



# ============================================================
# 5. Main
# ============================================================

def main():
    matched_ela, perf_df = load_matched_ela_rows()
    feature_cols = get_feature_cols(matched_ela)

    print("\nMatched ELA rows by source:")
    print(matched_ela["problem_type"].value_counts())
    print(f"\nInitial feature cols: {len(feature_cols)}")

    X_scaled, used_feature_cols = build_feature_matrix(matched_ela, feature_cols)

    print(f"Feature cols used after robust cleaning: {len(used_feature_cols)}")
    print(f"Cleaned matrix shape: {X_scaled.shape}")
    print(f"Cleaned matrix finite: {np.all(np.isfinite(X_scaled))}")

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_scaled)

    df_plot = matched_ela[["problem_type", "problem_name", "fid", "iid", "dim"]].copy()
    df_plot["PC1"] = Z[:, 0]
    df_plot["PC2"] = Z[:, 1]

    # Full plots and full coordinates.
    plot_pca_scatter(df_plot, pca, suffix="")
    plot_source_panels(df_plot, pca, suffix="")
    plot_bbob_highlight(df_plot, pca, suffix="")
    plot_pairwise_overlays(df_plot, pca, suffix="")
    dist_summary = compute_distance_summary(df_plot, X_scaled)
    save_basic_counts(df_plot, perf_df, feature_cols, used_feature_cols, pca)

    pca_table = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    pca_table.to_csv(os.path.join(OUT_DIR, "pca_explained_variance.csv"), index=False)

    df_plot.to_csv(os.path.join(OUT_DIR, "ela_space_pca_coordinates.csv"), index=False)

    # Filtered visualization only: removes extreme points so the main density is visible.
    df_plot_filtered, outliers_removed = filter_pca_outliers_for_plot(df_plot)
    if len(df_plot_filtered) > 0 and len(df_plot_filtered) < len(df_plot):
        plot_pca_scatter(df_plot_filtered, pca, suffix="_filtered")
        plot_source_panels(df_plot_filtered, pca, suffix="_filtered")
        plot_bbob_highlight(df_plot_filtered, pca, suffix="_filtered")
        plot_pairwise_overlays(df_plot_filtered, pca, suffix="_filtered")

    pd.DataFrame({"feature": used_feature_cols}).to_csv(
        os.path.join(OUT_DIR, "ela_features_used_after_cleaning.csv"),
        index=False,
    )

    print("\nPairwise nearest-neighbor distance summary:")
    print(dist_summary.sort_values(["source", "nn_distance_mean"]).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
