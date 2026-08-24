
# -*- coding: utf-8 -*-
"""
Visualize ELA-space distances among real-world, BBOB, MABBOB, and LLM problems.

This script uses matplotlib only for plotting.

It produces:
1. realworld_to_source_distance_boxplots.png
   Pairwise ELA-space distance distributions from each real-world problem
   to BBOB / MABBOB / LLM.

2. realworld_to_source_knn_heatmap.png
   k-nearest-neighbor mean distance from each real-world problem
   to each source.

3. intra_source_average_distances.png
   Internal average pairwise distances within BBOB / MABBOB / LLM / REAL.

4. pca_ela_distribution.png
   PCA visualization of actual distributions.

5. umap_ela_distribution.png
   UMAP visualization after optional PCA pre-reduction.

6. CSV files with numerical summaries.

LLM filtering:
--------------
LLM-generated problems can contain extreme ELA outliers.
This script optionally filters LLM rows using robust distance to the non-LLM
centroid in standardized ELA space.

Default:
    FILTER_LLM_OUTLIERS = True
    LLM_OUTLIER_QUANTILE = 0.98

This removes the most distant 2% of LLM problems according to robust
centroid distance. You can set FILTER_LLM_OUTLIERS = False to disable it.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. Paths and config
# ============================================================

BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"

OUT_DIR = "data/Combined/ela_space_distance_and_distribution"
os.makedirs(OUT_DIR, exist_ok=True)

DISTANCE_METRIC = "euclidean"  # "euclidean", "manhattan", or "cosine"
K_NEIGHBORS = 5

# Aggregate multiple ELA rows of the same real-world problem into one centroid.
AGGREGATE_REALWORLD_BY_PROBLEM = True

# If True, each BBOB fid / MABBOB iid / LLM iid is represented by one centroid.
# This usually gives cleaner distance and dimensionality-reduction plots.
AGGREGATE_SOURCES_BY_PROBLEM = True

# LLM outlier filtering.
FILTER_LLM_OUTLIERS = True
LLM_OUTLIER_QUANTILE = 0.98
LLM_OUTLIER_DISTANCE_MODE = "to_non_llm_centroid"  # currently supported mode

# Dimensionality reduction.
# PCA is kept for global linear structure.
# UMAP is used for nonlinear manifold visualization.
PCA_N_COMPONENTS_FOR_UMAP = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = 42

META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim",
    "seed", "n_samples", "instance_id", "mabbob_instance_id",
    "llm_problem_id", "selection_method",
    "lower_bound_min", "lower_bound_max",
    "upper_bound_min", "upper_bound_max", "source_dataset",
]


# ============================================================
# 2. Data loading
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


def numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def drop_invalid_rows(df, label):
    df = df.copy()
    n0 = len(df)

    if "FAILED" in df.columns:
        failed = numeric_series(df["FAILED"]).fillna(0) != 0
        df = df.loc[~failed].copy()

    if "dim" in df.columns:
        dim = numeric_series(df["dim"])
        valid = np.isfinite(dim) & (dim > 0)
        df = df.loc[valid].copy()
        df["dim"] = dim.loc[df.index].astype(int)

    if len(df) < n0:
        print(f"[Clean] {label}: kept {len(df)} / {n0} rows.")
    return df


def load_bbob_and_realworld():
    df = pd.read_csv(BBOB_ELA_PATH)
    df = harmonize_feature_names(df)
    df = drop_invalid_rows(df, "BBOB+REAL")

    if "fid" not in df.columns:
        raise ValueError("pipeline_aligned_ela.csv must contain fid.")

    fid = numeric_series(df["fid"])

    bbob = df.loc[np.isfinite(fid) & (fid >= 1) & (fid <= 24)].copy()
    bbob["fid"] = numeric_series(bbob["fid"]).astype(int)
    if "iid" not in bbob.columns:
        bbob["iid"] = 1
    bbob["iid"] = numeric_series(bbob["iid"]).fillna(1).astype(int)
    bbob["problem_type"] = "BBOB"
    bbob["problem_name"] = bbob["fid"].apply(lambda x: f"BBOB_F{int(x)}")

    real = df.loc[np.isfinite(fid) & (fid < 1)].copy()
    if real.empty:
        raise RuntimeError("No real-world ELA rows found with fid < 1.")

    if "iid" not in real.columns:
        real["iid"] = 1
    real["iid"] = numeric_series(real["iid"]).fillna(1).astype(int)
    real["problem_type"] = "REAL"

    if "problem_name" not in real.columns:
        real["problem_name"] = real["fid"].astype(str)
    real["problem_name"] = real["problem_name"].fillna(real["fid"].astype(str))

    return bbob, real


def load_mabbob():
    df = pd.read_csv(MABBOB_ELA_PATH)
    df = harmonize_feature_names(df)
    df = drop_invalid_rows(df, "MABBOB")

    if "mabbob_instance_id" not in df.columns:
        if "instance_id" in df.columns:
            df["mabbob_instance_id"] = df["instance_id"]
        elif "iid" in df.columns:
            df["mabbob_instance_id"] = df["iid"]
        else:
            raise ValueError("MABBOB ELA must contain mabbob_instance_id, instance_id, or iid.")

    iid = numeric_series(df["mabbob_instance_id"])
    df = df.loc[np.isfinite(iid)].copy()
    df["mabbob_instance_id"] = iid.loc[df.index].astype(int)

    df["problem_type"] = "MABBOB"
    df["fid"] = -100
    df["iid"] = df["mabbob_instance_id"].astype(int)
    df["problem_name"] = df["iid"].apply(lambda x: f"MABBOB_{int(x)}")
    return df


def load_llm():
    df = pd.read_csv(LLM_ELA_PATH)
    df = harmonize_feature_names(df)
    df = drop_invalid_rows(df, "LLM")

    if "llm_problem_id" not in df.columns:
        if "iid" in df.columns:
            df["llm_problem_id"] = df["iid"]
        elif "instance_id" in df.columns:
            df["llm_problem_id"] = df["instance_id"]
        else:
            raise ValueError("LLM ELA must contain llm_problem_id, iid, or instance_id.")

    iid = numeric_series(df["llm_problem_id"])
    df = df.loc[np.isfinite(iid)].copy()
    df["llm_problem_id"] = iid.loc[df.index].astype(int)

    df["problem_type"] = "LLM"
    df["fid"] = -200
    df["iid"] = df["llm_problem_id"].astype(int)

    if "problem_name" not in df.columns:
        df["problem_name"] = df["iid"].apply(lambda x: f"LLM_{int(x)}")
    df["problem_name"] = df["problem_name"].fillna(df["iid"].apply(lambda x: f"LLM_{int(x)}"))
    return df


# ============================================================
# 3. Feature matrix
# ============================================================

def get_feature_cols(all_df):
    excluded = set(META_COLS)
    cols = []

    for c in all_df.columns:
        if c in excluded:
            continue
        if c in ["FAILED", "ERROR"]:
            continue
        if c.endswith(".FAILED") or c.endswith(".ERROR"):
            continue
        if pd.api.types.is_numeric_dtype(all_df[c]):
            cols.append(c)

    X = all_df[cols].replace([np.inf, -np.inf], np.nan)
    cols = [c for c in cols if not X[c].isna().all()]

    if cols:
        nunique = X[cols].nunique(dropna=True)
        cols = [c for c in cols if nunique[c] > 1]

    return sorted(cols)


def clean_X(df, feature_cols):
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.astype(float)


def standardize_all(all_df, feature_cols):
    X = clean_X(all_df, feature_cols)
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    out = all_df.copy()
    for j, c in enumerate(feature_cols):
        out[c] = Z[:, j]

    return out, scaler


def aggregate_by_problem(df, feature_cols):
    rows = []

    for (ptype, pname), sub in df.groupby(["problem_type", "problem_name"], dropna=False):
        row = {
            "problem_type": ptype,
            "problem_name": pname,
            "n_rows": len(sub),
        }

        if "fid" in sub.columns:
            vals = numeric_series(sub["fid"]).dropna()
            row["fid"] = int(vals.iloc[0]) if len(vals) else np.nan
        if "iid" in sub.columns:
            vals = numeric_series(sub["iid"]).dropna()
            row["iid"] = int(vals.iloc[0]) if len(vals) else np.nan
        if "dim" in sub.columns:
            dims = sorted(numeric_series(sub["dim"]).dropna().astype(int).unique().tolist())
            row["dims"] = ",".join(map(str, dims))
        else:
            row["dims"] = ""

        for f in feature_cols:
            row[f] = sub[f].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_analysis_table(all_std, feature_cols):
    parts = []

    for ptype, sub in all_std.groupby("problem_type"):
        if ptype == "REAL":
            if AGGREGATE_REALWORLD_BY_PROBLEM:
                parts.append(aggregate_by_problem(sub, feature_cols))
            else:
                tmp = sub.copy()
                tmp["n_rows"] = 1
                tmp["dims"] = tmp["dim"].astype(str) if "dim" in tmp.columns else ""
                parts.append(tmp)
        else:
            if AGGREGATE_SOURCES_BY_PROBLEM:
                parts.append(aggregate_by_problem(sub, feature_cols))
            else:
                tmp = sub.copy()
                tmp["n_rows"] = 1
                tmp["dims"] = tmp["dim"].astype(str) if "dim" in tmp.columns else ""
                parts.append(tmp)

    analysis_df = pd.concat(parts, ignore_index=True, sort=False)
    return analysis_df


# ============================================================
# 4. LLM outlier filtering
# ============================================================

def filter_llm_outliers(analysis_df, feature_cols):
    if not FILTER_LLM_OUTLIERS:
        analysis_df["llm_outlier_distance"] = np.nan
        analysis_df["is_llm_outlier"] = False
        return analysis_df, pd.DataFrame()

    df = analysis_df.copy()
    non_llm = df[df["problem_type"] != "LLM"]
    llm = df[df["problem_type"] == "LLM"]

    if llm.empty:
        df["llm_outlier_distance"] = np.nan
        df["is_llm_outlier"] = False
        return df, pd.DataFrame()

    X_non = non_llm[feature_cols].to_numpy(dtype=float)
    centroid = X_non.mean(axis=0, keepdims=True)

    X_llm = llm[feature_cols].to_numpy(dtype=float)
    dist = pairwise_distances(X_llm, centroid, metric=DISTANCE_METRIC).ravel()

    threshold = float(np.quantile(dist, LLM_OUTLIER_QUANTILE))

    df["llm_outlier_distance"] = np.nan
    df["is_llm_outlier"] = False

    llm_index = llm.index.to_numpy()
    df.loc[llm_index, "llm_outlier_distance"] = dist
    df.loc[llm_index, "is_llm_outlier"] = dist > threshold

    outliers = df[(df["problem_type"] == "LLM") & (df["is_llm_outlier"])].copy()

    kept_df = df[~df["is_llm_outlier"]].copy()

    print(
        f"[LLM filter] Removed {len(outliers)} / {len(llm)} LLM problems "
        f"with distance > q{LLM_OUTLIER_QUANTILE} threshold {threshold:.4f}."
    )

    return kept_df, outliers


# ============================================================
# 5. Distance summaries
# ============================================================

def distance_matrix(df_a, df_b, feature_cols):
    Xa = df_a[feature_cols].to_numpy(dtype=float)
    Xb = df_b[feature_cols].to_numpy(dtype=float)
    return pairwise_distances(Xa, Xb, metric=DISTANCE_METRIC)


def summarize_real_to_sources(analysis_df, feature_cols):
    real = analysis_df[analysis_df["problem_type"] == "REAL"].copy()
    sources = ["BBOB", "MABBOB", "LLM"]

    rows = []
    pair_rows = []

    for _, rr in real.iterrows():
        real_one = pd.DataFrame([rr])
        for src in sources:
            ref = analysis_df[analysis_df["problem_type"] == src].copy()
            if ref.empty:
                continue

            d = distance_matrix(real_one, ref, feature_cols).ravel()
            k_eff = min(K_NEIGHBORS, len(d))
            d_sorted = np.sort(d)
            knn = float(np.mean(d_sorted[:k_eff]))

            source_centroid = ref[feature_cols].to_numpy(dtype=float).mean(axis=0, keepdims=True)
            d_centroid = float(pairwise_distances(
                real_one[feature_cols].to_numpy(dtype=float),
                source_centroid,
                metric=DISTANCE_METRIC,
            ).ravel()[0])

            rows.append({
                "real_problem": rr["problem_name"],
                "source": src,
                "n_source_problems": len(ref),
                "distance_min": float(np.min(d)),
                "distance_q25": float(np.quantile(d, 0.25)),
                "distance_median": float(np.median(d)),
                "distance_mean": float(np.mean(d)),
                "distance_q75": float(np.quantile(d, 0.75)),
                "distance_max": float(np.max(d)),
                f"distance_knn_mean_k{k_eff}": knn,
                "distance_to_source_centroid": d_centroid,
            })

            for ref_name, dist_val in zip(ref["problem_name"].tolist(), d):
                pair_rows.append({
                    "real_problem": rr["problem_name"],
                    "source": src,
                    "source_problem": ref_name,
                    "distance": float(dist_val),
                })

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def summarize_intra_source_distances(analysis_df, feature_cols):
    rows = []

    for src, sub in analysis_df.groupby("problem_type"):
        if len(sub) < 2:
            rows.append({
                "source": src,
                "n_problems": len(sub),
                "intra_distance_mean": np.nan,
                "intra_distance_median": np.nan,
                "intra_distance_min": np.nan,
                "intra_distance_max": np.nan,
            })
            continue

        X = sub[feature_cols].to_numpy(dtype=float)
        D = pairwise_distances(X, X, metric=DISTANCE_METRIC)

        tri = D[np.triu_indices_from(D, k=1)]

        rows.append({
            "source": src,
            "n_problems": len(sub),
            "intra_distance_mean": float(np.mean(tri)),
            "intra_distance_median": float(np.median(tri)),
            "intra_distance_min": float(np.min(tri)),
            "intra_distance_q25": float(np.quantile(tri, 0.25)),
            "intra_distance_q75": float(np.quantile(tri, 0.75)),
            "intra_distance_max": float(np.max(tri)),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. Plots
# ============================================================

def plot_real_to_source_boxplots(pair_df, out_path):
    sources = ["BBOB", "MABBOB", "LLM"]
    real_names = list(pair_df["real_problem"].drop_duplicates())

    fig, ax = plt.subplots(figsize=(max(12, 2.5 * len(real_names)), 7))

    data = []
    positions = []
    xticks = []
    xticklabels = []

    group_gap = 1.0
    offset = 0.28

    for i, real_name in enumerate(real_names):
        center = 1.0 + i * (len(sources) * offset + group_gap)
        xticks.append(center + offset)
        xticklabels.append(real_name)

        for j, src in enumerate(sources):
            vals = pair_df[
                (pair_df["real_problem"] == real_name) &
                (pair_df["source"] == src)
            ]["distance"].values

            if len(vals) == 0:
                continue

            positions.append(center + j * offset)
            data.append(vals)

    bp = ax.boxplot(data, positions=positions, widths=0.22, showfliers=False, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=25, ha="right")
    ax.set_ylabel(f"{DISTANCE_METRIC} distance in standardized ELA space")
    ax.set_title("Distance from each real-world problem to BBOB / MABBOB / LLM")
    ax.grid(axis="y", alpha=0.25)

    # Manual legend by box order color cycle is not stable, so use text note.
    note = "Within each real-world group, box order: BBOB, MABBOB, LLM"
    ax.text(0.01, 0.98, note, transform=ax.transAxes, va="top", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_knn_heatmap(summary_df, out_path):
    sources = ["BBOB", "MABBOB", "LLM"]
    real_names = list(summary_df["real_problem"].drop_duplicates())
    val_col = f"distance_knn_mean_k{K_NEIGHBORS}"

    mat = np.full((len(real_names), len(sources)), np.nan)

    for i, real_name in enumerate(real_names):
        for j, src in enumerate(sources):
            sub = summary_df[
                (summary_df["real_problem"] == real_name) &
                (summary_df["source"] == src)
            ]
            if len(sub):
                # if source size < K, column name may be k_eff instead of K.
                possible_cols = [c for c in sub.columns if c.startswith("distance_knn_mean_k")]
                col = val_col if val_col in sub.columns else possible_cols[0]
                mat[i, j] = float(sub[col].iloc[0])

    fig, ax = plt.subplots(figsize=(7, max(4, 0.65 * len(real_names))))
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources)
    ax.set_yticks(range(len(real_names)))
    ax.set_yticklabels(real_names)
    ax.set_title(f"Mean distance to {K_NEIGHBORS}-nearest neighbors")

    for i in range(len(real_names)):
        for j in range(len(sources)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{DISTANCE_METRIC} distance")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_intra_source_bar(intra_df, out_path):
    plot_df = intra_df.copy()
    order = ["REAL", "BBOB", "MABBOB", "LLM"]
    plot_df["order"] = plot_df["source"].map({s: i for i, s in enumerate(order)})
    plot_df = plot_df.sort_values("order")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["source"], plot_df["intra_distance_mean"])
    ax.set_ylabel(f"Mean internal pairwise distance ({DISTANCE_METRIC})")
    ax.set_title("Internal average ELA-space distance within each problem set")
    ax.grid(axis="y", alpha=0.25)

    for i, row in enumerate(plot_df.itertuples()):
        val = row.intra_distance_mean
        if np.isfinite(val):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pca_distribution(analysis_df, feature_cols, out_path):
    X = analysis_df[feature_cols].to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    plot_df = analysis_df.copy()
    plot_df["PC1"] = Z[:, 0]
    plot_df["PC2"] = Z[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))

    markers = {
        "REAL": "*",
        "BBOB": "o",
        "MABBOB": "s",
        "LLM": "^",
    }

    for src in ["BBOB", "MABBOB", "LLM", "REAL"]:
        sub = plot_df[plot_df["problem_type"] == src]
        if sub.empty:
            continue

        size = 110 if src == "REAL" else 32
        alpha = 0.95 if src == "REAL" else 0.55

        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            marker=markers.get(src, "o"),
            s=size,
            alpha=alpha,
            label=f"{src} (n={len(sub)})",
        )

        if src == "REAL":
            for _, r in sub.iterrows():
                ax.text(r["PC1"], r["PC2"], str(r["problem_name"]), fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    ax.set_title("PCA projection of ELA feature space")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    coord_path = out_path.replace(".png", "_coordinates.csv")
    plot_df[["problem_type", "problem_name", "PC1", "PC2"]].to_csv(coord_path, index=False)
    print(f"Saved: {coord_path}")



def plot_umap_distribution(analysis_df, feature_cols, out_path):
    try:
        import umap
    except Exception as e:
        raise ImportError(
            "UMAP requires the package 'umap-learn'. Install it with:\n"
            "    pip install umap-learn\n"
            "or:\n"
            "    conda install -c conda-forge umap-learn"
        ) from e

    X = analysis_df[feature_cols].to_numpy(dtype=float)
    n = len(X)

    if n < 5:
        print("[UMAP] Too few points. Skipping.")
        return

    n_pca = min(PCA_N_COMPONENTS_FOR_UMAP, X.shape[1], max(2, n - 1))
    X_pre = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    Z = reducer.fit_transform(X_pre)

    plot_df = analysis_df.copy()
    plot_df["UMAP1"] = Z[:, 0]
    plot_df["UMAP2"] = Z[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))

    markers = {
        "REAL": "*",
        "BBOB": "o",
        "MABBOB": "s",
        "LLM": "^",
    }

    for src in ["BBOB", "MABBOB", "LLM", "REAL"]:
        sub = plot_df[plot_df["problem_type"] == src]
        if sub.empty:
            continue

        size = 110 if src == "REAL" else 32
        alpha = 0.95 if src == "REAL" else 0.55

        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            marker=markers.get(src, "o"),
            s=size,
            alpha=alpha,
            label=f"{src} (n={len(sub)})",
        )

        if src == "REAL":
            for _, r in sub.iterrows():
                ax.text(r["UMAP1"], r["UMAP2"], str(r["problem_name"]), fontsize=8)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"UMAP projection of ELA feature space "
        f"(n_neighbors={n_neighbors}, min_dist={UMAP_MIN_DIST})"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    coord_path = out_path.replace(".png", "_coordinates.csv")
    plot_df[["problem_type", "problem_name", "UMAP1", "UMAP2"]].to_csv(coord_path, index=False)
    print(f"Saved: {coord_path}")


# ============================================================
# 7. Main
# ============================================================

def main():
    print("Loading ELA data...")
    bbob, real = load_bbob_and_realworld()
    mabbob = load_mabbob()
    llm = load_llm()

    raw_all = pd.concat([bbob, real, mabbob, llm], ignore_index=True, sort=False)

    feature_cols = get_feature_cols(raw_all)
    if not feature_cols:
        raise RuntimeError("No valid ELA feature columns found.")

    print(f"Using {len(feature_cols)} ELA features.")

    all_std, scaler = standardize_all(raw_all, feature_cols)
    analysis_df = prepare_analysis_table(all_std, feature_cols)

    before_counts = analysis_df["problem_type"].value_counts().to_dict()
    analysis_df, llm_outliers = filter_llm_outliers(analysis_df, feature_cols)
    after_counts = analysis_df["problem_type"].value_counts().to_dict()

    analysis_df.to_csv(os.path.join(OUT_DIR, "analysis_points_after_llm_filtering.csv"), index=False)
    llm_outliers.to_csv(os.path.join(OUT_DIR, "filtered_llm_outliers.csv"), index=False)

    print("Counts before filtering:", before_counts)
    print("Counts after filtering:", after_counts)

    # Distance summaries
    summary_df, pair_df = summarize_real_to_sources(analysis_df, feature_cols)

    summary_path = os.path.join(OUT_DIR, "realworld_to_source_distance_summary.csv")
    pair_path = os.path.join(OUT_DIR, "realworld_to_source_pairwise_distances.csv")

    summary_df.to_csv(summary_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {pair_path}")

    # Internal average distances
    intra_df = summarize_intra_source_distances(analysis_df, feature_cols)
    intra_path = os.path.join(OUT_DIR, "intra_source_average_distances.csv")
    intra_df.to_csv(intra_path, index=False)
    print(f"Saved: {intra_path}")

    # Plots
    plot_real_to_source_boxplots(
        pair_df,
        os.path.join(OUT_DIR, "realworld_to_source_distance_boxplots.png"),
    )
    plot_knn_heatmap(
        summary_df,
        os.path.join(OUT_DIR, "realworld_to_source_knn_heatmap.png"),
    )
    plot_intra_source_bar(
        intra_df,
        os.path.join(OUT_DIR, "intra_source_average_distances.png"),
    )
    plot_pca_distribution(
        analysis_df,
        feature_cols,
        os.path.join(OUT_DIR, "pca_ela_distribution.png"),
    )
    plot_umap_distribution(
        analysis_df,
        feature_cols,
        os.path.join(OUT_DIR, "umap_ela_distribution.png"),
    )

    # Save feature list and config.
    with open(os.path.join(OUT_DIR, "ela_feature_cols_used.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    config = {
        "distance_metric": DISTANCE_METRIC,
        "k_neighbors": K_NEIGHBORS,
        "aggregate_realworld_by_problem": AGGREGATE_REALWORLD_BY_PROBLEM,
        "aggregate_sources_by_problem": AGGREGATE_SOURCES_BY_PROBLEM,
        "filter_llm_outliers": FILTER_LLM_OUTLIERS,
        "llm_outlier_quantile": LLM_OUTLIER_QUANTILE,
        "llm_outlier_distance_mode": LLM_OUTLIER_DISTANCE_MODE,
        "pca_n_components_for_umap": PCA_N_COMPONENTS_FOR_UMAP,
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
        "umap_metric": UMAP_METRIC,
        "n_features": len(feature_cols),
        "counts_before_filtering": before_counts,
        "counts_after_filtering": after_counts,
    }
    with open(os.path.join(OUT_DIR, "plot_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
