import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment


MABBOB_RAW_ELA = "mabbob_ela_results/mabbob_ela_raw.csv"

OUT_DIR = "data/MABBOB"
os.makedirs(OUT_DIR, exist_ok=True)

SELECT_METHOD = "lhs_target"  # "lhs_target" or "farthest"
N_SELECT = 240                # 建议先 240，对应 BBOB 24 * 10
PCA_DIM = 5
RANDOM_SEED = 42


def harmonize_mabbob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    尽量把之前 MA-BBOB ELA 脚本的列名，对齐到你原 pipeline 的命名习惯。

    你原来的 BBOB/real-world pipeline 使用:
      ela_distr, disp, ic 等命名
    我之前给你的 MA-BBOB raw 脚本里可能使用:
      ela_distribution, dispersion, information_content

    这里做一个轻量映射。
    """
    rename = {}
    for c in df.columns:
        nc = c
        nc = nc.replace("ela_distribution.", "ela_distr.")
        nc = nc.replace("dispersion.", "disp.")
        nc = nc.replace("information_content.", "ic.")
        rename[c] = nc
    df = df.rename(columns=rename)

    if "problem_type" not in df.columns:
        df["problem_type"] = "MA-BBOB"

    if "fid" not in df.columns:
        # 用负数 fid 避免和 BBOB 1..24 冲突
        df["fid"] = -100

    if "problem_name" not in df.columns:
        df["problem_name"] = df["instance_id"].apply(
            lambda x: f"MABBOB_{int(x)}"
        )

    if "seed" not in df.columns:
        # 你的 pipeline 以 seed 作为 meta column，这里简单设为 instance_id
        df["seed"] = df["instance_id"]

    return df


def get_feature_matrix(df: pd.DataFrame):
    meta_cols = [
        "problem_type",
        "problem_name",
        "fid",
        "iid",
        "instance_id",
        "dim",
        "seed",
        "n_samples",
    ]

    Xdf = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")
    Xdf = Xdf.select_dtypes(include=[np.number])
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    # 删除全空、常数、FAILED 指示列
    Xdf = Xdf.dropna(axis=1, how="all")
    failed_cols = [c for c in Xdf.columns if c.endswith(".FAILED")]
    Xdf = Xdf.drop(columns=failed_cols, errors="ignore")

    nunique = Xdf.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    Xdf = Xdf.drop(columns=const_cols, errors="ignore")

    X_imp = SimpleImputer(strategy="median").fit_transform(Xdf)
    X_std = StandardScaler().fit_transform(X_imp)

    return Xdf.columns.tolist(), X_std


def farthest_point_selection(Z, n_select, seed=0):
    rng = np.random.default_rng(seed)
    n = Z.shape[0]

    selected = [int(rng.integers(n))]
    dist_to_selected = pairwise_distances(Z, Z[selected]).reshape(-1)

    for _ in range(1, n_select):
        idx = int(np.argmax(dist_to_selected))
        selected.append(idx)
        new_dist = pairwise_distances(Z, Z[[idx]]).reshape(-1)
        dist_to_selected = np.minimum(dist_to_selected, new_dist)

    return np.array(selected, dtype=int)


def lhs_target_selection(Z, n_select, seed=0):
    """
    目标：在 ELA embedding 空间中构造近似均匀的 LHS target points，
    然后为每个 target 匹配最近的 MA-BBOB candidate。
    """
    Z01 = MinMaxScaler().fit_transform(Z)

    sampler = qmc.LatinHypercube(d=Z01.shape[1], seed=seed)
    targets = sampler.random(n=n_select)

    D = pairwise_distances(targets, Z01)
    row_ind, col_ind = linear_sum_assignment(D)

    return np.array(col_ind[:n_select], dtype=int)


def main():
    if not os.path.exists(MABBOB_RAW_ELA):
        raise FileNotFoundError(
            f"Cannot find {MABBOB_RAW_ELA}. "
            "Please run the MA-BBOB ELA generation script first."
        )

    df = pd.read_csv(MABBOB_RAW_ELA)
    df = harmonize_mabbob_columns(df)

    if "iid" not in df.columns:
        # 后续训练和 merge 用 iid 作为 instance key
        df["iid"] = df["instance_id"].astype(int)

    feature_cols, X_std = get_feature_matrix(df)

    pca_dim = min(PCA_DIM, X_std.shape[1], X_std.shape[0] - 1)
    Z = PCA(n_components=pca_dim, random_state=RANDOM_SEED).fit_transform(X_std)

    n_select = min(N_SELECT, len(df))

    if SELECT_METHOD == "farthest":
        idx = farthest_point_selection(Z, n_select, seed=RANDOM_SEED)
    elif SELECT_METHOD == "lhs_target":
        idx = lhs_target_selection(Z, n_select, seed=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown SELECT_METHOD: {SELECT_METHOD}")

    selected = df.iloc[idx].copy()
    selected["selection_method"] = SELECT_METHOD
    selected["mabbob_instance_id"] = selected["instance_id"].astype(int)
    selected["fid"] = -100
    selected["iid"] = selected["mabbob_instance_id"]

    selected_instances = selected[
        ["problem_name", "fid", "iid", "dim", "seed",
         "mabbob_instance_id", "selection_method"]
    ].copy()

    selected_instances_path = os.path.join(OUT_DIR, "selected_mabbob_instances.csv")
    selected_ela_path = os.path.join(OUT_DIR, "mabbob_selected_ela.csv")

    selected_instances.to_csv(selected_instances_path, index=False)
    selected.to_csv(selected_ela_path, index=False)

    print(f"[Saved] {selected_instances_path}")
    print(f"[Saved] {selected_ela_path}")
    print(f"Selected {len(selected)} MA-BBOB instances with method={SELECT_METHOD}.")


if __name__ == "__main__":
    main()