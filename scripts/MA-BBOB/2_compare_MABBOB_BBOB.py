import pflacco.classical_ela_features as ela
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import qmc
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ioh
import os
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 参数
# ============================================================

DIM = 5

# 原始 BBOB: 24 个 function，每个 function 取多少个 instance
BBOB_FUNCTION_IDS = list(range(1, 25))
BBOB_INSTANCE_IDS = list(range(1, 6))  # 可改成 range(1, 16)

# ELA 采样数量
N_SAMPLES_PER_INSTANCE = DIM * 200
# 正式实验可以改成 DIM * 1000

LOWER_BOUND = -5
UPPER_BOUND = 5
RANDOM_SEED = 42

OUTDIR = "mabbob_ela_results"
os.makedirs(OUTDIR, exist_ok=True)

MABBOB_RAW_PATH = os.path.join(OUTDIR, "mabbob_ela_raw.csv")
BBOB_RAW_PATH = os.path.join(OUTDIR, "bbob_ela_raw.csv")


# ============================================================
# 2. Sobol 采样
# ============================================================

def sobol_sample(dim, n, seed=0, lower=-5, upper=5):
    m = int(np.ceil(np.log2(n)))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    X01 = sampler.random_base2(m=m)
    X01 = X01[:n]
    X = qmc.scale(X01, lower, upper)
    return X


# ============================================================
# 3. 获取 BBOB problem
# ============================================================

def get_bbob_problem(function_id, instance_id, dim):
    """
    获取原始 BBOB problem。

    不同 ioh 版本的 API 可能略有差异，所以这里用了 fallback。
    """
    try:
        return ioh.get_problem(
            function_id,
            instance=instance_id,
            dimension=dim,
            problem_class=ioh.ProblemClass.BBOB,
        )
    except Exception:
        try:
            return ioh.get_problem(function_id, instance_id, dim)
        except Exception:
            return ioh.problem.RealSingleObjective.BBOB(function_id, instance_id, dim)


# ============================================================
# 4. ELA 计算函数
# ============================================================

def evaluate_problem_on_sample(problem, X):
    y = np.array([problem(x) for x in X], dtype=float)
    return y


def safe_add_features(feature_dict, prefix, func, *args, **kwargs):
    try:
        feats = func(*args, **kwargs)
        for k, v in feats.items():
            feature_dict[f"{prefix}.{k}"] = v
    except Exception as e:
        feature_dict[f"{prefix}.FAILED"] = 1
        feature_dict[f"{prefix}.ERROR"] = str(e)
    return feature_dict


def clean_y(y):
    y = np.asarray(y, dtype=float)

    finite_mask = np.isfinite(y)

    if not np.any(finite_mask):
        return np.zeros_like(y)

    finite_y = y[finite_mask]

    y = np.nan_to_num(
        y,
        nan=np.nanmedian(finite_y),
        posinf=np.max(finite_y),
        neginf=np.min(finite_y),
    )

    # log 压缩，减弱函数值尺度差异
    y_shift = y - np.min(y)
    y_trans = np.log1p(y_shift)

    return y_trans


def compute_ela_features(problem, dim, problem_type, function_id, instance_id, n_samples, seed):
    X = sobol_sample(
        dim=dim,
        n=n_samples,
        seed=seed,
        lower=LOWER_BOUND,
        upper=UPPER_BOUND,
    )

    y_raw = evaluate_problem_on_sample(problem, X)
    y_trans = clean_y(y_raw)

    feats = {
        "problem_type": problem_type,
        "function_id": function_id,
        "instance_id": instance_id,
        "dim": dim,
        "n_samples": n_samples,
        "y_min": float(np.min(y_raw)),
        "y_max": float(np.max(y_raw)),
        "y_mean": float(np.mean(y_raw)),
        "y_std": float(np.std(y_raw)),
    }

    feats = safe_add_features(
        feats, "ela_meta", ela.calculate_ela_meta, X, y_trans)
    feats = safe_add_features(
        feats, "ela_distribution", ela.calculate_ela_distribution, X, y_trans)
    feats = safe_add_features(
        feats, "ela_level", ela.calculate_ela_level, X, y_trans)
    feats = safe_add_features(feats, "pca", ela.calculate_pca, X, y_trans)

    feats = safe_add_features(
        feats,
        "limo",
        ela.calculate_limo,
        X,
        y_trans,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        force=True,
    )

    feats = safe_add_features(
        feats, "nbc", ela.calculate_nbc, X, y_trans, minimize=True)
    feats = safe_add_features(
        feats, "dispersion", ela.calculate_dispersion, X, y_trans, minimize=True)

    feats = safe_add_features(
        feats,
        "information_content",
        ela.calculate_information_content,
        X,
        y_trans,
        seed=seed,
    )

    return feats


# ============================================================
# 5. 计算原始 BBOB ELA
# ============================================================

def generate_bbob_ela():
    if os.path.exists(BBOB_RAW_PATH):
        print(f"[Load] Existing BBOB ELA file: {BBOB_RAW_PATH}")
        return pd.read_csv(BBOB_RAW_PATH)

    rows = []

    total = len(BBOB_FUNCTION_IDS) * len(BBOB_INSTANCE_IDS)

    with tqdm(total=total, desc="Computing ELA for original BBOB") as pbar:
        for fid in BBOB_FUNCTION_IDS:
            for iid in BBOB_INSTANCE_IDS:
                problem = get_bbob_problem(
                    function_id=fid,
                    instance_id=iid,
                    dim=DIM,
                )

                feats = compute_ela_features(
                    problem=problem,
                    dim=DIM,
                    problem_type="BBOB",
                    function_id=fid,
                    instance_id=iid,
                    n_samples=N_SAMPLES_PER_INSTANCE,
                    seed=RANDOM_SEED + 1000 * fid + iid,
                )

                rows.append(feats)
                pbar.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(BBOB_RAW_PATH, index=False)

    print(f"[Saved] BBOB ELA features: {BBOB_RAW_PATH}")
    return df


# ============================================================
# 6. 读取 MA-BBOB ELA
# ============================================================

def load_mabbob_ela():
    if not os.path.exists(MABBOB_RAW_PATH):
        raise FileNotFoundError(
            f"Cannot find {MABBOB_RAW_PATH}. "
            f"Please run run_mabbob_ela_experiment.py first."
        )

    df = pd.read_csv(MABBOB_RAW_PATH)

    # 如果旧脚本里没有 problem_type / function_id，则补上
    if "problem_type" not in df.columns:
        df["problem_type"] = "MA-BBOB"

    if "function_id" not in df.columns:
        df["function_id"] = -1

    return df


# ============================================================
# 7. 合并两个数据集并做 PCA
# ============================================================

def prepare_combined_pca(df_bbob, df_mabbob):
    df = pd.concat([df_bbob, df_mabbob], axis=0, ignore_index=True)

    meta_cols = [
        "problem_type",
        "function_id",
        "instance_id",
        "dim",
        "n_samples",
    ]

    numeric_df = df.drop(
        columns=[c for c in meta_cols if c in df.columns], errors="ignore")
    numeric_df = numeric_df.select_dtypes(include=[np.number])

    # 删除 FAILED 指示列
    failed_cols = [c for c in numeric_df.columns if c.endswith(".FAILED")]
    numeric_df = numeric_df.drop(columns=failed_cols, errors="ignore")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # 删除几乎常数的列
    nunique = numeric_df.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    numeric_df = numeric_df.drop(columns=constant_cols, errors="ignore")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(numeric_df)
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_scaled)

    zdf = df[meta_cols].copy()
    zdf["pc1"] = Z[:, 0]
    zdf["pc2"] = Z[:, 1]

    zdf["pca_explained_var_ratio_pc1"] = pca.explained_variance_ratio_[0]
    zdf["pca_explained_var_ratio_pc2"] = pca.explained_variance_ratio_[1]

    out_path = os.path.join(OUTDIR, "bbob_vs_mabbob_ela_pca.csv")
    zdf.to_csv(out_path, index=False)

    print(f"[Saved] Combined PCA coordinates: {out_path}")
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Number of ELA features used:", numeric_df.shape[1])

    return df, numeric_df, X_scaled, Z, zdf, pca


# ============================================================
# 8. Coverage metrics
# ============================================================

def coverage_metrics(Z_group, Z_reference):
    """
    Z_group: 某一类问题，例如 BBOB 或 MA-BBOB
    Z_reference: 全部候选点，用于计算覆盖全部空间的 nearest distance
    """
    if len(Z_group) < 2:
        return {
            "n": len(Z_group),
            "mean_pairwise_dist": np.nan,
            "min_pairwise_dist": np.nan,
            "mean_dist_reference_to_nearest": np.nan,
            "max_dist_reference_to_nearest": np.nan,
        }

    D = pairwise_distances(Z_group, Z_group)
    upper = D[np.triu_indices_from(D, k=1)]

    D_ref = pairwise_distances(Z_reference, Z_group)
    nearest = D_ref.min(axis=1)

    return {
        "n": len(Z_group),
        "mean_pairwise_dist": float(np.mean(upper)),
        "min_pairwise_dist": float(np.min(upper)),
        "mean_dist_reference_to_nearest": float(np.mean(nearest)),
        "max_dist_reference_to_nearest": float(np.max(nearest)),
    }


def compute_group_coverage(zdf):
    Z_all = zdf[["pc1", "pc2"]].values

    rows = []
    for group in sorted(zdf["problem_type"].unique()):
        mask = zdf["problem_type"].values == group
        Z_group = Z_all[mask]

        rows.append({
            "problem_type": group,
            **coverage_metrics(Z_group, Z_all),
        })

    metrics = pd.DataFrame(rows)

    out_path = os.path.join(OUTDIR, "bbob_vs_mabbob_coverage_metrics.csv")
    metrics.to_csv(out_path, index=False)

    print(f"[Saved] Coverage metrics: {out_path}")
    print(metrics)

    return metrics


# ============================================================
# 9. 作图
# ============================================================

def plot_scatter(zdf):
    plt.figure(figsize=(8, 6))

    for group in ["BBOB", "MA-BBOB"]:
        sub = zdf[zdf["problem_type"] == group]
        plt.scatter(
            sub["pc1"],
            sub["pc2"],
            s=35 if group == "BBOB" else 20,
            alpha=0.8 if group == "BBOB" else 0.35,
            label=f"{group} (n={len(sub)})",
        )

    plt.xlabel("ELA PCA 1")
    plt.ylabel("ELA PCA 2")
    plt.title("ELA space comparison: original BBOB vs MA-BBOB")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTDIR, "bbob_vs_mabbob_pca_scatter.png")
    plt.savefig(out_path, dpi=250)
    plt.close()

    print(f"[Saved] {out_path}")


def plot_by_bbob_function(zdf):
    """
    只给 BBOB 点标记 function_id，MA-BBOB 作为背景。
    这个图适合看 MA-BBOB 是否扩展到了原始 24 个函数之外的区域。
    """
    bbob = zdf[zdf["problem_type"] == "BBOB"]
    mabbob = zdf[zdf["problem_type"] == "MA-BBOB"]

    plt.figure(figsize=(9, 7))

    plt.scatter(
        mabbob["pc1"],
        mabbob["pc2"],
        s=18,
        alpha=0.20,
        label="MA-BBOB background",
    )

    sc = plt.scatter(
        bbob["pc1"],
        bbob["pc2"],
        c=bbob["function_id"],
        s=55,
        alpha=0.9,
        label="BBOB",
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("BBOB function id")

    plt.xlabel("ELA PCA 1")
    plt.ylabel("ELA PCA 2")
    plt.title("BBOB functions over MA-BBOB ELA space")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUTDIR, "bbob_functions_over_mabbob_pca.png")
    plt.savefig(out_path, dpi=250)
    plt.close()

    print(f"[Saved] {out_path}")


def plot_density_like(zdf):
    """
    不用 seaborn，仅用 matplotlib hexbin 画分布密度。
    """
    for group in ["BBOB", "MA-BBOB"]:
        sub = zdf[zdf["problem_type"] == group]

        plt.figure(figsize=(8, 6))
        plt.hexbin(
            sub["pc1"],
            sub["pc2"],
            gridsize=25,
            mincnt=1,
        )

        plt.xlabel("ELA PCA 1")
        plt.ylabel("ELA PCA 2")
        plt.title(f"ELA PCA density: {group}")
        plt.colorbar(label="count")
        plt.tight_layout()

        safe_group = group.lower().replace("-", "")
        out_path = os.path.join(OUTDIR, f"{safe_group}_ela_pca_hexbin.png")
        plt.savefig(out_path, dpi=250)
        plt.close()

        print(f"[Saved] {out_path}")


# ============================================================
# 10. 主程序
# ============================================================

def main():
    print("Loading MA-BBOB ELA...")
    df_mabbob = load_mabbob_ela()

    print("Computing or loading original BBOB ELA...")
    df_bbob = generate_bbob_ela()

    print("Preparing combined ELA-PCA space...")
    df, numeric_df, X_scaled, Z, zdf, pca = prepare_combined_pca(
        df_bbob=df_bbob,
        df_mabbob=df_mabbob,
    )

    print("Computing coverage metrics...")
    metrics = compute_group_coverage(zdf)

    print("Plotting...")
    plot_scatter(zdf)
    plot_by_bbob_function(zdf)
    plot_density_like(zdf)

    print("\nDone.")
    print(f"Results are saved in: {OUTDIR}")
    print("\nMain figures:")
    print("1. bbob_vs_mabbob_pca_scatter.png")
    print("2. bbob_functions_over_mabbob_pca.png")
    print("3. bbob_ela_pca_hexbin.png")
    print("4. mabbob_ela_pca_hexbin.png")
    print("\nMain CSV:")
    print("1. bbob_vs_mabbob_ela_pca.csv")
    print("2. bbob_vs_mabbob_coverage_metrics.csv")


if __name__ == "__main__":
    main()
