
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


BBOB_ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
BBOB_PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"

MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
MABBOB_PERF_PATH = "data/MABBOB/mabbob_algorithm_auc_performance.csv"

LLM_ELA_PATH = "data/LLM/llm_generated_ela.csv"
LLM_PERF_PATH = "data/LLM/llm_algorithm_auc_performance.csv"

OUT_DIR = "results/ela_space_compare/"
os.makedirs(OUT_DIR, exist_ok=True)

ONLY_REGRESSOR_USED_INSTANCES = True
COLOR_BY = "fid"  # "fid" or "iid"
ZOOM_TO_BBOB = True
SAVE_GLOBAL_AXIS_VERSION = True

CLIP_LOW_Q = 0.001
CLIP_HIGH_Q = 0.999
ABS_CLIP = 1e100

META_COLS = [
    "problem_type", "problem_name", "fid", "iid", "dim", "seed", "n_samples",
    "instance_id", "mabbob_instance_id", "llm_problem_id", "selection_method",
    "lower_bound_min", "lower_bound_max", "upper_bound_min", "upper_bound_max",
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

    dim_num = pd.to_numeric(df["dim"], errors="coerce")
    valid_dim = np.isfinite(dim_num) & (dim_num > 0)
    df = df.loc[valid_dim].copy()
    df["dim"] = dim_num.loc[df.index].astype(int)

    n_drop = n0 - len(df)
    if n_drop > 0:
        print(f"[Clean] Dropped {n_drop} invalid {problem_type} ELA rows; kept {len(df)} / {n0}.")

    return df


def ensure_problem_keys(df, problem_type):
    df = harmonize_feature_names(df.copy())
    df = drop_invalid_ela_rows(df, problem_type)
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
        df = df.loc[np.isfinite(id_num)].copy()
        df["llm_problem_id"] = id_num.loc[df.index].astype(int)
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


def load_all_matched_ela_rows():
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

    if ONLY_REGRESSOR_USED_INSTANCES:
        perf_keys = perf_df[["problem_type", "fid", "iid", "dim"]].drop_duplicates()
        ela_df = pd.merge(
            ela_df,
            perf_keys,
            on=["problem_type", "fid", "iid", "dim"],
            how="inner",
        )

    ela_df = ela_df.drop_duplicates(
        subset=["problem_type", "fid", "iid", "dim", "problem_name"]
    ).reset_index(drop=True)

    return ela_df


def robust_log_signed_transform(X):
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

        lo = finite.quantile(CLIP_LOW_Q)
        hi = finite.quantile(CLIP_HIGH_Q)

        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = finite.min()
            hi = finite.max()

        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            X[c] = s.clip(lower=lo, upper=hi)

    X = X.clip(lower=-ABS_CLIP, upper=ABS_CLIP)
    X = np.sign(X) * np.log1p(np.abs(X))
    X = pd.DataFrame(X, columns=X.columns, index=X.index)
    X = X.replace([np.inf, -np.inf], np.nan)

    if dropped_cols:
        print(f"[Clean] Dropped {len(dropped_cols)} empty/constant/pathological feature cols before global PCA.")

    return X


def fit_global_map_and_transform(ela_df, feature_cols):
    X_raw = ela_df[feature_cols].copy()
    X_trans = robust_log_signed_transform(X_raw)
    used_feature_cols = X_trans.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_trans)
    X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25, 75))
    X_scaled = scaler.fit_transform(X_imp)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = np.clip(X_scaled, -1e6, 1e6)

    if not np.all(np.isfinite(X_scaled)):
        raise RuntimeError("Global cleaned matrix still contains non-finite values.")

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_scaled)

    return Z, pca, used_feature_cols


def plot_bbob_only_on_global_map(df_all, pca, suffix, zoom_to_bbob=True):
    bbob = df_all[df_all["problem_type"] == "BBOB"].copy()
    if bbob.empty:
        raise RuntimeError("No BBOB rows found after global mapping.")

    fig, ax = plt.subplots(figsize=(10, 8))

    if COLOR_BY == "fid":
        groups = sorted(bbob["fid"].unique())
        cmap = plt.cm.get_cmap("tab20", max(len(groups), 1))
        for i, fid in enumerate(groups):
            sub = bbob[bbob["fid"] == fid]
            ax.scatter(
                sub["PC1"], sub["PC2"],
                s=46, alpha=0.88, marker="o",
                edgecolors="black", linewidths=0.25,
                color=cmap(i % cmap.N),
                label=f"F{int(fid)}",
            )
        ax.legend(title="BBOB fid", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)

    elif COLOR_BY == "iid":
        groups = sorted(bbob["iid"].unique())
        cmap = plt.cm.get_cmap("tab10", max(len(groups), 1))
        for i, iid in enumerate(groups):
            sub = bbob[bbob["iid"] == iid]
            ax.scatter(
                sub["PC1"], sub["PC2"],
                s=46, alpha=0.88, marker="o",
                edgecolors="black", linewidths=0.25,
                color=cmap(i % cmap.N),
                label=f"iid={int(iid)}",
            )
        ax.legend(title="BBOB iid", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)

    if zoom_to_bbob:
        x = bbob["PC1"].to_numpy()
        y = bbob["PC2"].to_numpy()
        x_margin = 0.08 * (x.max() - x.min() + 1e-12)
        y_margin = 0.08 * (y.max() - y.min() + 1e-12)
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
        title_suffix = "zoomed to BBOB range"
    else:
        x = df_all["PC1"].to_numpy()
        y = df_all["PC2"].to_numpy()
        x_margin = 0.05 * (x.max() - x.min() + 1e-12)
        y_margin = 0.05 * (y.max() - y.min() + 1e-12)
        ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
        ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
        title_suffix = "global axis range"

    ax.set_xlabel(f"Global PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"Global PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title(f"BBOB only on the global ELA-PCA map\n({title_suffix})")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"bbob_only_on_global_pca_map_{suffix}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def plot_bbob_global_map_by_fid_panels(df_all, pca):
    bbob = df_all[df_all["problem_type"] == "BBOB"].copy()
    fids = sorted(bbob["fid"].unique())

    n_cols = 4
    n_rows = int(np.ceil(len(fids) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.5 * n_rows),
        sharex=True, sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    x_all = bbob["PC1"].to_numpy()
    y_all = bbob["PC2"].to_numpy()
    x_margin = 0.08 * (x_all.max() - x_all.min() + 1e-12)
    y_margin = 0.08 * (y_all.max() - y_all.min() + 1e-12)

    for ax, fid in zip(axes, fids):
        sub = bbob[bbob["fid"] == fid]
        ax.scatter(
            sub["PC1"], sub["PC2"],
            s=34, alpha=0.88,
            edgecolors="black", linewidths=0.25,
        )
        ax.set_title(f"F{int(fid)} | n={len(sub)}")
        ax.grid(alpha=0.25)
        ax.set_xlim(x_all.min() - x_margin, x_all.max() + x_margin)
        ax.set_ylim(y_all.min() - y_margin, y_all.max() + y_margin)

    for ax in axes[len(fids):]:
        ax.axis("off")

    fig.supxlabel(f"Global PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    fig.supylabel(f"Global PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    fig.suptitle("BBOB only on global ELA-PCA map by fid", y=1.01)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "bbob_only_on_global_pca_map_by_fid_panels.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    ela_df = load_all_matched_ela_rows()
    feature_cols = get_feature_cols(ela_df)

    print("Matched ELA rows by source:")
    print(ela_df["problem_type"].value_counts())
    print(f"Initial feature cols: {len(feature_cols)}")

    Z, pca, used_feature_cols = fit_global_map_and_transform(ela_df, feature_cols)

    df_all = ela_df[["problem_type", "problem_name", "fid", "iid", "dim"]].copy()
    if "seed" in ela_df.columns:
        df_all["seed"] = ela_df["seed"].values

    df_all["PC1"] = Z[:, 0]
    df_all["PC2"] = Z[:, 1]

    df_all.to_csv(os.path.join(OUT_DIR, "global_pca_coordinates_all_sources.csv"), index=False)
    df_all[df_all["problem_type"] == "BBOB"].to_csv(
        os.path.join(OUT_DIR, "bbob_only_on_global_pca_map_coordinates.csv"),
        index=False,
    )

    pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }).to_csv(os.path.join(OUT_DIR, "global_pca_map_explained_variance.csv"), index=False)

    pd.DataFrame({"feature": used_feature_cols}).to_csv(
        os.path.join(OUT_DIR, "global_pca_map_features_used_after_cleaning.csv"),
        index=False,
    )

    plot_bbob_only_on_global_map(df_all, pca, suffix=f"by_{COLOR_BY}_zoomed", zoom_to_bbob=True)

    if SAVE_GLOBAL_AXIS_VERSION:
        plot_bbob_only_on_global_map(df_all, pca, suffix=f"by_{COLOR_BY}_global_axis", zoom_to_bbob=False)

    plot_bbob_global_map_by_fid_panels(df_all, pca)

    print("Done.")


if __name__ == "__main__":
    main()
