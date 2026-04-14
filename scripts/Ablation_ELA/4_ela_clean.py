# fmt: off
import os
import sys
import numpy as np
import pandas as pd
# from scipy.stats import spearmanr
import warnings
sys.path.insert(0, os.getcwd())
from utils.problems_factory import ProblemName
from utils.calculate_single_ela_feature_set import ela_sets
# fmt: on


def process_and_clean_ela(problem_name, target_n_coef=1000,
                          target_block_coef=0.0, seed_range=100):
    print(f"\n--- Processing Problem: {problem_name} ---")
    all_seeds_data = []
    for seed in range(seed_range):
        seed_features = {'seed': seed}
        for ela_set_val in range(8):
            file_name = f"ProblemName.{problem_name}-{ela_set_val}-seed:{seed}-block_coef:{target_block_coef}-n_coef:{target_n_coef}.csv"
            file_path = f"data/Ablation_ELA/atom/{file_name}"
            # print(os.path.exists(file_path))
            # print(file_path)

            if os.path.exists(file_path):
                try:
                    df_atom = pd.read_csv(file_path)
                    cols_to_keep = [c for c in df_atom.columns if c not in [
                        'n_coef', 'block_coef', 'seed']]
                    feature_dict = df_atom[cols_to_keep].iloc[0].to_dict()
                    seed_features.update(feature_dict)
                except Exception as e:
                    pass

        if len(seed_features) > 1:
            all_seeds_data.append(seed_features)

    if not all_seeds_data:
        print(f"  [!] No valid data found for {problem_name}. Skipping.")
        return None

    df_raw = pd.DataFrame(all_seeds_data)
    df_raw.set_index('seed', inplace=True)
    initial_feat_count = df_raw.shape[1]
    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned = df_raw.dropna(axis=1, thresh=int(0.8 * len(df_raw)))
    df_cleaned = df_cleaned.fillna(df_cleaned.mean())
    # df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() > 1]
    basic_cleaned_count = df_cleaned.shape[1]

    corr_matrix = df_cleaned.corr(method='spearman').abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(
        upper_tri[column] > 0.90)]
    df_final = df_cleaned.drop(columns=to_drop)

    final_feat_count = df_final.shape[1]

    print(f"  -> Extracted features : {initial_feat_count}")
    print(
        f"  -> After basic clean  : {basic_cleaned_count} (removed NaNs/Constants)")
    print(
        f"  -> After collinearity : {final_feat_count} (removed {len(to_drop)} highly correlated)")

    return df_final


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    OUTPUT_DIR = "data/Ablation_ELA/Processed_ELA_Phase1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    problems = [p.value for p in ProblemName] if hasattr(
        ProblemName, '__iter__') else ["P1", "P2"]
    try:
        problems_list = [name for name,
                         member in ProblemName.__members__.items()]
    except:
        problems_list = ["F1", "F2", "F3"]

    for prob in problems_list:
        df_processed = process_and_clean_ela(
            prob, target_n_coef=1000, target_block_coef=0.0)

        if df_processed is not None and not df_processed.empty:
            save_path = f"{OUTPUT_DIR}/cleaned_ela_{prob}.csv"
            df_processed.to_csv(save_path)
            print(f"  [+] Saved cleaned data to {save_path}")

    print("\n  [+]Phase 1: Feature processing and collinearity cleaning completed.")
