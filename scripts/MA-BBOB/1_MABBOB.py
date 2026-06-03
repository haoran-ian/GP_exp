import os
import warnings
warnings.filterwarnings("ignore")

import ioh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import qmc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import pflacco.classical_ela_features as ela


# ============================================================
# 1. 全局参数
# ============================================================

DIM = 5
N_CANDIDATES = 1000
N_SELECT = 100
N_SAMPLES_PER_INSTANCE = DIM * 1000

LOWER_BOUND = -5
UPPER_BOUND = 5

RANDOM_SEED = 42

OUTDIR = "mabbob_ela_results"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# 2. 采样函数：Sobol sample in [-5, 5]^d
# ============================================================

def sobol_sample(dim, n, seed=0, lower=-5, upper=5):
    """
    Sobol sample. Sobol 对样本数最好是 2 的幂；
    这里为了方便，把 n 向上取到最近的 2^m，然后截断。
    """
    m = int(np.ceil(np.log2(n)))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    X01 = sampler.random_base2(m=m)
    X01 = X01[:n]
    X = qmc.scale(X01, lower, upper)
    return X


# ============================================================
# 3. 生成 MA-BBOB problem
# ============================================================

def get_mabbob_problem(instance_id, dim):
    """
    IOH 内置 MA-BBOB。
    instance_id 控制随机生成过程，因此可复现。
    """
    return ioh.problem.ManyAffine(instance_id, n_variables=dim)


# ============================================================
# 4. 计算一个 problem 的 ELA 特征
# ============================================================

def evaluate_problem_on_sample(problem, X):
    """
    逐点调用 IOH problem。
    注意：IOH problem 通常是最小化问题。
    """
    y = np.array([problem(x) for x in X], dtype=float)
    return y


def safe_add_features(feature_dict, prefix, func, *args, **kwargs):
    """
    某些 ELA 特征可能因为样本数、数值问题失败。
    为了让实验不中断，这里失败时跳过，并记录 warning 字段。
    """
    try:
        feats = func(*args, **kwargs)
        for k, v in feats.items():
            feature_dict[f"{prefix}.{k}"] = v
    except Exception as e:
        feature_dict[f"{prefix}.FAILED"] = 1
        feature_dict[f"{prefix}.ERROR"] = str(e)
    return feature_dict


def compute_ela_features(problem, dim, instance_id, n_samples, seed):
    """
    计算一个 MA-BBOB 实例的 ELA 特征。
    """
    X = sobol_sample(
        dim=dim,
        n=n_samples,
        seed=seed,
        lower=LOWER_BOUND,
        upper=UPPER_BOUND,
    )

    y = evaluate_problem_on_sample(problem, X)

    # 数值清洗：有些函数值可能非常大
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=np.nanmedian(y), posinf=np.nanmax(y[np.isfinite(y)]), neginf=np.nanmin(y[np.isfinite(y)]))

    # 对 y 做 log 压缩，减少尺度差异
    # MA-BBOB/BBOB 的函数值尺度可能差异很大，log1p 通常更稳
    y_shift = y - np.min(y)
    y_trans = np.log1p(y_shift)

    feats = {
        "instance_id": instance_id,
        "dim": dim,
        "n_samples": n_samples,
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
    }

    # 这些是不需要额外调用目标函数的 classical ELA 特征
    feats = safe_add_features(feats, "ela_meta", ela.calculate_ela_meta, X, y_trans)
    feats = safe_add_features(feats, "ela_distribution", ela.calculate_ela_distribution, X, y_trans)
    feats = safe_add_features(feats, "ela_level", ela.calculate_ela_level, X, y_trans)
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
    feats = safe_add_features(feats, "nbc", ela.calculate_nbc, X, y_trans, minimize=True)
    feats = safe_add_features(feats, "dispersion", ela.calculate_dispersion, X, y_trans, minimize=True)
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
# 5. 候选池生成
# ============================================================

def generate_candidate_pool():
    """
    生成 N_CANDIDATES 个 MA-BBOB 实例，并计算 ELA。
    """
    rows = []

    for instance_id in tqdm(range(1, N_CANDIDATES + 1), desc="Computing ELA for MA-BBOB candidates"):
        problem = get_mabbob_problem(instance_id=instance_id, dim=DIM)

        feats = compute_ela_features(
            problem=problem,
            dim=DIM,
            instance_id=instance_id,
            n_samples=N_SAMPLES_PER_INSTANCE,
            seed=RANDOM_SEED + instance_id,
        )

        rows.append(feats)

    df = pd.DataFrame(rows)

    raw_path = os.path.join(OUTDIR, "mabbob_ela_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"[Saved] Raw ELA features: {raw_path}")

    return df


# ============================================================
# 6. ELA 特征清洗 + PCA
# ============================================================

def prepare_feature_matrix(df):
    """
    从原始 ELA dataframe 中取出数值特征，清洗，标准化，PCA 到 2D。
    """
    meta_cols = ["instance_id", "dim", "n_samples"]

    # 删除错误文本列
    numeric_df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
    numeric_df = numeric_df.select_dtypes(include=[np.number])

    # 删除 FAILED 指示列也可以；这里保留也没关系，但为了干净先删掉
    failed_cols = [c for c in numeric_df.columns if c.endswith(".FAILED")]
    numeric_df = numeric_df.drop(columns=failed_cols, errors="ignore")

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    # 删除全 NaN 列
    numeric_df = numeric_df.dropna(axis=1, how="all")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(numeric_df)
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_scaled)

    zdf = pd.DataFrame({
        "instance_id": df["instance_id"].values,
        "pc1": Z[:, 0],
        "pc2": Z[:, 1],
    })

    zdf["pca_explained_var_ratio_pc1"] = pca.explained_variance_ratio_[0]
    zdf["pca_explained_var_ratio_pc2"] = pca.explained_variance_ratio_[1]

    pca_path = os.path.join(OUTDIR, "mabbob_ela_pca.csv")
    zdf.to_csv(pca_path, index=False)
    print(f"[Saved] PCA coordinates: {pca_path}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    return numeric_df, X_scaled, Z, zdf


# ============================================================
# 7. 两种实例选择方法
# ============================================================

def random_selection(n_total, n_select, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_select, replace=False)


def farthest_point_selection(Z, n_select, seed=0):
    """
    在 ELA-PCA 空间里做 farthest point sampling。
    目标：选出来的点彼此尽量远。
    """
    rng = np.random.default_rng(seed)
    n_total = Z.shape[0]

    first = rng.integers(n_total)
    selected = [first]

    dist_to_selected = pairwise_distances(Z, Z[[first]]).reshape(-1)

    for _ in range(1, n_select):
        next_idx = int(np.argmax(dist_to_selected))
        selected.append(next_idx)

        new_dist = pairwise_distances(Z, Z[[next_idx]]).reshape(-1)
        dist_to_selected = np.minimum(dist_to_selected, new_dist)

    return np.array(selected, dtype=int)


# ============================================================
# 8. 覆盖度指标
# ============================================================

def coverage_metrics(Z, selected_indices):
    """
    简单覆盖度指标：
    1. 选中点之间的平均 pairwise distance
    2. 选中点之间的最小 pairwise distance
    3. 所有候选点到最近选中点的平均距离
    4. 所有候选点到最近选中点的最大距离
    """
    Z_sel = Z[selected_indices]

    D_sel = pairwise_distances(Z_sel, Z_sel)
    upper = D_sel[np.triu_indices_from(D_sel, k=1)]

    D_all_to_sel = pairwise_distances(Z, Z_sel)
    nearest = D_all_to_sel.min(axis=1)

    return {
        "mean_pairwise_dist_selected": float(np.mean(upper)),
        "min_pairwise_dist_selected": float(np.min(upper)),
        "mean_dist_all_to_nearest_selected": float(np.mean(nearest)),
        "max_dist_all_to_nearest_selected": float(np.max(nearest)),
    }


def compare_selection_methods(Z, zdf):
    n_total = Z.shape[0]

    random_idx = random_selection(
        n_total=n_total,
        n_select=N_SELECT,
        seed=RANDOM_SEED,
    )

    farthest_idx = farthest_point_selection(
        Z=Z,
        n_select=N_SELECT,
        seed=RANDOM_SEED,
    )

    selected_random = zdf.iloc[random_idx].copy()
    selected_random["method"] = "random"

    selected_farthest = zdf.iloc[farthest_idx].copy()
    selected_farthest["method"] = "farthest"

    selected = pd.concat([selected_random, selected_farthest], axis=0)

    selected_path = os.path.join(OUTDIR, "selected_instances.csv")
    selected.to_csv(selected_path, index=False)
    print(f"[Saved] Selected instances: {selected_path}")

    metrics = pd.DataFrame([
        {"method": "random", **coverage_metrics(Z, random_idx)},
        {"method": "farthest", **coverage_metrics(Z, farthest_idx)},
    ])

    metrics_path = os.path.join(OUTDIR, "coverage_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"[Saved] Coverage metrics: {metrics_path}")

    print("\nCoverage metrics:")
    print(metrics)

    return random_idx, farthest_idx, metrics


# ============================================================
# 9. 画图
# ============================================================

def plot_selection(Z, zdf, random_idx, farthest_idx):
    plt.figure(figsize=(7, 6))
    plt.scatter(zdf["pc1"], zdf["pc2"], s=20, alpha=0.25, label="all candidates")
    plt.scatter(zdf.iloc[random_idx]["pc1"], zdf.iloc[random_idx]["pc2"], s=60, marker="x", label="random selected")
    plt.xlabel("ELA PCA 1")
    plt.ylabel("ELA PCA 2")
    plt.title("Random selection in ELA space")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTDIR, "random_selection_pca.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[Saved] {path}")

    plt.figure(figsize=(7, 6))
    plt.scatter(zdf["pc1"], zdf["pc2"], s=20, alpha=0.25, label="all candidates")
    plt.scatter(zdf.iloc[farthest_idx]["pc1"], zdf.iloc[farthest_idx]["pc2"], s=60, marker="x", label="farthest selected")
    plt.xlabel("ELA PCA 1")
    plt.ylabel("ELA PCA 2")
    plt.title("Farthest-point selection in ELA space")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTDIR, "farthest_selection_pca.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[Saved] {path}")


# ============================================================
# 10. 主程序
# ============================================================

def main():
    df = generate_candidate_pool()
    numeric_df, X_scaled, Z, zdf = prepare_feature_matrix(df)
    random_idx, farthest_idx, metrics = compare_selection_methods(Z, zdf)
    plot_selection(Z, zdf, random_idx, farthest_idx)

    print("\nDone.")
    print(f"Results are in folder: {OUTDIR}")
    print("\n你下一步可以打开：")
    print("1. mabbob_ela_results/mabbob_ela_raw.csv")
    print("2. mabbob_ela_results/mabbob_ela_pca.csv")
    print("3. mabbob_ela_results/selected_instances.csv")
    print("4. mabbob_ela_results/coverage_metrics.csv")
    print("5. mabbob_ela_results/random_selection_pca.png")
    print("6. mabbob_ela_results/farthest_selection_pca.png")


if __name__ == "__main__":
    main()