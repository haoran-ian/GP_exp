
# -*- coding: utf-8 -*-
"""
Plot distances in ELA feature space between each real-world problem and
BBOB / MABBOB / LLM problem sets, using matplotlib only.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"

OUT_DIR = "data/Combined/ela_distance_realworld_vs_bbob_mabbob_llm"
os.makedirs(OUT_DIR, exist_ok=True)

DISTANCE_METRIC = "euclidean"
K_NEIGHBORS = 5
AGGREGATE_REALWORLD_BY_PROBLEM = True

META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim",
    "seed", "n_samples", "instance_id", "mabbob_instance_id",
    "llm_problem_id", "selection_method",
    "lower_bound_min", "lower_bound_max",
    "upper_bound_min", "upper_bound_max", "source_dataset",
]


def harmonize_feature_names(df):
    rename = {}
    for c in df.columns:
        nc = c
        nc = nc.replace("ela_distribution.", "ela_distr.")
        nc = nc.replace("dispersion.", "disp.")
        nc = nc.replace("information_content.", "ic.")
        rename[c] = nc
    return df.rename(columns=rename)


def safe_int_series(s):
    return pd.to_numeric(s, errors="coerce")


def drop_invalid_rows(df, problem_type, stage="ELA"):
    df = df.copy()
    n0 = len(df)

    if "FAILED" in df.columns:
        failed_mask = pd.to_numeric(df["FAILED"], errors="coerce").fillna(0) != 0
        df = df.loc[~failed_mask].copy()

    if "dim" in df.columns:
        dim_num = pd.to_numeric(df["dim"], errors="coerce")
        valid_dim = np.isfinite(dim_num) & (dim_num > 0)
        df = df.loc[valid_dim].copy()
        df["dim"] = dim_num.loc[df.index].astype(int)

    n_drop = n0 - len(df)
    if n_drop > 0:
        print(f"[Clean] Dropped {n_drop} invalid {problem_type} {stage} rows; kept {len(df)} / {n0}.")
    return df


def load_bbob_and_realworld():
    df = pd.read_csv(BBOB_ELA_PATH)
    df = harmonize_feature_names(df)
    df = drop_invalid_rows(df, "BBOB_REAL")

    if "fid" not in df.columns:
        raise ValueError("BBOB/real-world ELA file must contain 'fid'.")

    fid_num = safe_int_series(df["fid"])

    bbob_df = df.loc[np.isfinite(fid_num) & (fid_num >= 1) & (fid_num <= 24)].copy()
    bbob_df["fid"] = safe_int_series(bbob_df["fid"]).astype(int)
    if "iid" not in bbob_df.columns:
        bbob_df["iid"] = 1
    bbob_df["iid"] = safe_int_series(bbob_df["iid"]).fillna(1).astype(int)
    bbob_df["problem_type"] = "BBOB"
    if "problem_name" not in bbob_df.columns:
        bbob_df["problem_name"] = bbob_df["fid"].apply(lambda x: f"BBOB_F{int(x)}")
    bbob_df["problem_name"] = bbob_df["problem_name"].fillna(
        bbob_df["fid"].apply(lambda x: f"BBOB_F{int(x)}")
    )

    real_df = df.loc[np.isfinite(fid_num) & (fid_num < 1)].copy()
    if real_df.empty:
        raise RuntimeError("No real-world rows found with fid < 1.")
    if "iid" not in real_df.columns:
        real_df["iid"] = 1
    real_df["iid"] = safe_int_series(real_df["iid"]).fillna(1).astype(int)
    if "problem_name" not in real_df.columns:
        real_df["problem_name"] = [f"REAL_{i}" for i in range(len(real_df))]
    real_df["problem_type"] = "REAL"

    return bbob_df, real_df


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

    id_num = safe_int_series(df["mabbob_instance_id"])
    df = df.loc[np.isfinite(id_num)].copy()
    df["mabbob_instance_id"] = id_num.loc[df.index].astype(int)

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

    id_num = safe_int_series(df["llm_problem_id"])
    df = df.loc[np.isfinite(id_num)].copy()
    df["llm_problem_id"] = id_num.loc[df.index].astype(int)

    df["problem_type"] = "LLM"
    df["fid"] = -200
    df["iid"] = df["llm_problem_id"].astype(int)
    if "problem_name" not in df.columns:
        df["problem_name"] = df["iid"].apply(lambda x: f"LLM_{int(x)}")
    df["problem_name"] = df["problem_name"].fillna(df["iid"].apply(lambda x: f"LLM_{int(x)}"))
    return df


def get_feature_cols(all_df):
    excluded = set(META_COLS)
    feature_cols = []
    for c in all_df.columns:
        if c in excluded:
            continue
        if c.endswith(".FAILED") or c.endswith(".ERROR") or c in ["FAILED", "ERROR"]:
            continue
        if pd.api.types.is_numeric_dtype(all_df[c]):
            feature_cols.append(c)

    X = all_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in feature_cols if not X[c].isna().all()]

    if feature_cols:
        nunique = X[feature_cols].nunique(dropna=True)
        const_cols = nunique[nunique <= 1].index.tolist()
        feature_cols = [c for c in feature_cols if c not in const_cols]

    return sorted(feature_cols)


def clean_X(df, feature_cols):
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.astype(float)


def attach_standardized_features(df, feature_cols, scaler=None):
    X = clean_X(df, feature_cols)
    if scaler is None:
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
    else:
        Z = scaler.transform(X)

    Z_df = pd.DataFrame(Z, index=df.index, columns=feature_cols)
    out = df.copy()
    for c in feature_cols:
        out[c] = Z_df[c]
    return out, scaler


def make_realworld_representation(real_df, feature_cols):
    if AGGREGATE_REALWORLD_BY_PROBLEM:
        rows = []
        for name, sub in real_df.groupby("problem_name"):
            row = {"problem_name": name, "n_rows": len(sub)}
            if "dim" in sub.columns:
                dims = sorted(pd.to_numeric(sub["dim"], errors="coerce").dropna().astype(int).unique().tolist())
                row["dims"] = ",".join(map(str, dims))
            else:
                row["dims"] = ""
            for f in feature_cols:
                row[f] = sub[f].mean()
            rows.append(row)
        return pd.DataFrame(rows)
    else:
        out = real_df.copy()
        out["n_rows"] = 1
        out["dims"] = out["dim"].astype(str) if "dim" in out.columns else ""
        return out


def compute_distance_summaries(real_repr, ref_df, feature_cols, source_name, k=5, metric="euclidean"):
    rows = []
    X_ref = ref_df[feature_cols].to_numpy(dtype=float)
    source_centroid = X_ref.mean(axis=0, keepdims=True)

    for _, rr in real_repr.iterrows():
        x = rr[feature_cols].to_numpy(dtype=float).reshape(1, -1)
        dists = pairwise_distances(x, X_ref, metric=metric).ravel()
        k_eff = min(k, len(dists))
        d_sorted = np.sort(dists)
        knn_mean = float(np.mean(d_sorted[:k_eff])) if k_eff > 0 else np.nan
        centroid_dist = float(pairwise_distances(x, source_centroid, metric=metric).ravel()[0])

        rows.append({
            "real_problem": rr["problem_name"],
            "source": source_name,
            "n_real_rows_aggregated": int(rr["n_rows"]) if "n_rows" in rr else 1,
            "dims": rr.get("dims", ""),
            "n_ref_instances": int(len(ref_df)),
            "distance_min": float(np.min(dists)),
            "distance_q25": float(np.quantile(dists, 0.25)),
            "distance_median": float(np.median(dists)),
            "distance_mean": float(np.mean(dists)),
            "distance_q75": float(np.quantile(dists, 0.75)),
            "distance_max": float(np.max(dists)),
            f"distance_knn_mean_k{k_eff}": knn_mean,
            "distance_to_source_centroid": centroid_dist,
        })
    return pd.DataFrame(rows)


def compute_all_pair_dist_for_plot(real_repr, ref_df, feature_cols, source_name, metric="euclidean"):
    rows = []
    X_ref = ref_df[feature_cols].to_numpy(dtype=float)
    for _, rr in real_repr.iterrows():
        x = rr[feature_cols].to_numpy(dtype=float).reshape(1, -1)
        dists = pairwise_distances(x, X_ref, metric=metric).ravel()
        for d in dists:
            rows.append({
                "real_problem": rr["problem_name"],
                "source": source_name,
                "distance": float(d),
            })
    return pd.DataFrame(rows)


def plot_distance_boxplots(all_pair_df, out_path):
    sources = ["BBOB", "MABBOB", "LLM"]
    real_problems = list(all_pair_df["real_problem"].drop_duplicates())

    fig, ax = plt.subplots(figsize=(max(12, 2.8 * len(real_problems)), 7))
    colors = {"BBOB": "#1f77b4", "MABBOB": "#ff7f0e", "LLM": "#2ca02c"}

    positions, data, box_colors = [], [], []
    xticks, xticklabels = [], []
    gap = 1.0
    offset = 0.32
    start = 1.0

    for i, rp in enumerate(real_problems):
        center = start + i * (len(sources) + gap)
        xticks.append(center + offset)
        xticklabels.append(rp)

        for j, src in enumerate(sources):
            pos = center + j * offset
            vals = all_pair_df[
                (all_pair_df["real_problem"] == rp) &
                (all_pair_df["source"] == src)
            ]["distance"].values
            if len(vals) == 0:
                continue
            positions.append(pos)
            data.append(vals)
            box_colors.append(colors[src])

    bp = ax.boxplot(
        data, positions=positions, widths=0.24, showfliers=False, patch_artist=True
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_ylabel(f"Pairwise {DISTANCE_METRIC} distance in standardized ELA space")
    ax.set_title("Distance from each real-world problem to BBOB / MABBOB / LLM instances")
    ax.grid(axis="y", alpha=0.25)
    for src in sources:
        ax.plot([], [], color=colors[src], linewidth=8, alpha=0.55, label=src)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_knn_heatmap(summary_df, out_path, k):
    sources = ["BBOB", "MABBOB", "LLM"]
    real_problems = list(summary_df["real_problem"].drop_duplicates())
    val_col = f"distance_knn_mean_k{k}"

    mat = np.full((len(real_problems), len(sources)), np.nan, dtype=float)
    for i, rp in enumerate(real_problems):
        for j, src in enumerate(sources):
            sub = summary_df[
                (summary_df["real_problem"] == rp) &
                (summary_df["source"] == src)
            ]
            if len(sub) > 0:
                mat[i, j] = float(sub[val_col].iloc[0])

    fig, ax = plt.subplots(figsize=(7, max(4, 0.7 * len(real_problems))))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources)
    ax.set_yticks(range(len(real_problems)))
    ax.set_yticklabels(real_problems)
    ax.set_title(f"Mean distance to {k}-nearest neighbors in ELA space")
    for i in range(len(real_problems)):
        for j in range(len(sources)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{DISTANCE_METRIC} distance")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_centroid_distance_bar(summary_df, out_path):
    sources = ["BBOB", "MABBOB", "LLM"]
    real_problems = list(summary_df["real_problem"].drop_duplicates())
    colors = {"BBOB": "#1f77b4", "MABBOB": "#ff7f0e", "LLM": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(max(10, 2.6 * len(real_problems)), 6))
    bar_width = 0.22
    x = np.arange(len(real_problems))

    for j, src in enumerate(sources):
        vals = []
        for rp in real_problems:
            sub = summary_df[
                (summary_df["real_problem"] == rp) &
                (summary_df["source"] == src)
            ]
            vals.append(float(sub["distance_to_source_centroid"].iloc[0]) if len(sub) > 0 else np.nan)
        ax.bar(x + j * bar_width - bar_width, vals, width=bar_width, label=src, alpha=0.75, color=colors[src])

    ax.set_xticks(x)
    ax.set_xticklabels(real_problems, rotation=20, ha="right")
    ax.set_ylabel(f"Distance to source centroid ({DISTANCE_METRIC})")
    ax.set_title("Real-world problem distance to source-set centroid in ELA space")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    bbob_df, real_df = load_bbob_and_realworld()
    mabbob_df = load_mabbob()
    llm_df = load_llm()

    all_df = pd.concat([bbob_df, real_df, mabbob_df, llm_df], ignore_index=True, sort=False)
    feature_cols = get_feature_cols(all_df)
    if len(feature_cols) == 0:
        raise RuntimeError("No valid numeric ELA features found.")
    print(f"Using {len(feature_cols)} ELA features.")

    all_df_std, scaler = attach_standardized_features(all_df, feature_cols, scaler=None)

    bbob_std = all_df_std[all_df_std["problem_type"] == "BBOB"].copy()
    real_std = all_df_std[all_df_std["problem_type"] == "REAL"].copy()
    mabbob_std = all_df_std[all_df_std["problem_type"] == "MABBOB"].copy()
    llm_std = all_df_std[all_df_std["problem_type"] == "LLM"].copy()

    real_repr = make_realworld_representation(real_std, feature_cols)

    print(f"Real-world problems: {len(real_repr)}")
    print(f"BBOB instances: {len(bbob_std)}")
    print(f"MABBOB instances: {len(mabbob_std)}")
    print(f"LLM instances: {len(llm_std)}")

    sum_bbob = compute_distance_summaries(real_repr, bbob_std, feature_cols, "BBOB", k=K_NEIGHBORS, metric=DISTANCE_METRIC)
    sum_mabbob = compute_distance_summaries(real_repr, mabbob_std, feature_cols, "MABBOB", k=K_NEIGHBORS, metric=DISTANCE_METRIC)
    sum_llm = compute_distance_summaries(real_repr, llm_std, feature_cols, "LLM", k=K_NEIGHBORS, metric=DISTANCE_METRIC)

    summary_df = pd.concat([sum_bbob, sum_mabbob, sum_llm], ignore_index=True)
    summary_path = os.path.join(OUT_DIR, "realworld_to_source_distance_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    pair_bbob = compute_all_pair_dist_for_plot(real_repr, bbob_std, feature_cols, "BBOB", metric=DISTANCE_METRIC)
    pair_mabbob = compute_all_pair_dist_for_plot(real_repr, mabbob_std, feature_cols, "MABBOB", metric=DISTANCE_METRIC)
    pair_llm = compute_all_pair_dist_for_plot(real_repr, llm_std, feature_cols, "LLM", metric=DISTANCE_METRIC)
    pair_df = pd.concat([pair_bbob, pair_mabbob, pair_llm], ignore_index=True)

    pair_path = os.path.join(OUT_DIR, "realworld_to_source_all_pair_distances.csv")
    pair_df.to_csv(pair_path, index=False)
    print(f"Saved: {pair_path}")

    plot_distance_boxplots(pair_df, os.path.join(OUT_DIR, "realworld_to_source_distance_boxplots.png"))
    plot_knn_heatmap(summary_df, os.path.join(OUT_DIR, "realworld_to_source_knn_heatmap.png"), k=K_NEIGHBORS)
    plot_centroid_distance_bar(summary_df, os.path.join(OUT_DIR, "realworld_to_source_centroid_distance_bar.png"))

    with open(os.path.join(OUT_DIR, "plot_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "distance_metric": DISTANCE_METRIC,
            "k_neighbors": K_NEIGHBORS,
            "aggregate_realworld_by_problem": AGGREGATE_REALWORLD_BY_PROBLEM,
            "n_features": len(feature_cols),
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, "ela_feature_cols_used.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
