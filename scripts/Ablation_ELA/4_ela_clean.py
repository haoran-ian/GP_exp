# fmt: off
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings

sys.path.insert(0, os.getcwd())
# 假设你依然有这些 utils
from utils.problems_factory import ProblemName
from utils.calculate_single_ela_feature_set import ela_sets
# fmt: on

def process_and_clean_ela(problem_name, target_n_coef=1000, target_block_coef=0.0, seed_range=100):
    """
    读取特定问题的全量 ELA 数据，拼接、清理、去共线性，并返回干净的 DataFrame
    """
    print(f"\n--- Processing Problem: {problem_name} ---")
    
    # 1. 提取与拼接数据
    all_seeds_data = []
    
    for seed in range(seed_range):
        seed_features = {'seed': seed}
        for ela_name, ela_set_val in ela_sets.items():
            file_name = f"{problem_name}-{ela_set_val}-seed:{seed}-block_coef:{target_block_coef}-n_coef:{target_n_coef}.csv"
            file_path = f"data/Ablation_ELA/atom/{file_name}"
            
            if os.path.exists(file_path):
                try:
                    # 假设单文件只有一行数据，或者读取第一行特征
                    df_atom = pd.read_csv(file_path)
                    
                    # 提取特征（如果前几列是 meta_data，视实际情况跳过，这里假设纯特征或字典化）
                    # 把该 ELA 类别下的所有特征转为字典加入当前 seed
                    # 排除非特征列如 'n_coef', 'block_coef', 'seed' 等
                    cols_to_keep = [c for c in df_atom.columns if c not in ['n_coef', 'block_coef', 'seed']]
                    feature_dict = df_atom[cols_to_keep].iloc[0].to_dict()
                    seed_features.update(feature_dict)
                except Exception as e:
                    pass # 跳过损坏的文件
                    
        if len(seed_features) > 1: # 说明不仅仅只有 'seed' 键
            all_seeds_data.append(seed_features)
            
    if not all_seeds_data:
        print(f"  [!] No valid data found for {problem_name}. Skipping.")
        return None

    # 转为 DataFrame，行为 seed (100次独立运行)，列为所有提取出来的 ELA 特征
    df_raw = pd.DataFrame(all_seeds_data)
    df_raw.set_index('seed', inplace=True)
    initial_feat_count = df_raw.shape[1]
    
    # 2. 基础清洗：处理 NaN, Inf 和 常数特征
    # 替换 Inf 为 NaN
    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 剔除缺失值过多的特征 (例如 > 20% 的 seed 都计算失败的特征)
    df_cleaned = df_raw.dropna(axis=1, thresh=int(0.8 * len(df_raw)))
    
    # 用列均值填充剩余的少量 NaN
    df_cleaned = df_cleaned.fillna(df_cleaned.mean())
    
    # 剔除方差为 0 的常数特征 (在所有 seed 上表现一致，无信息量)
    df_cleaned = df_cleaned.loc[:, df_cleaned.nunique() > 1]
    
    basic_cleaned_count = df_cleaned.shape[1]
    
    # 3. 消除多重共线性 (Spearman correlation > 0.9)
    # 计算特征间的 Spearman 相关系数矩阵的绝对值
    corr_matrix = df_cleaned.corr(method='spearman').abs()
    
    # 获取相关系数矩阵的上三角部分
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找出所有与之前特征相关性大于 0.9 的冗余特征
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
    df_final = df_cleaned.drop(columns=to_drop)
    
    final_feat_count = df_final.shape[1]
    
    print(f"  -> Extracted features : {initial_feat_count}")
    print(f"  -> After basic clean  : {basic_cleaned_count} (removed NaNs/Constants)")
    print(f"  -> After collinearity : {final_feat_count} (removed {len(to_drop)} highly correlated)")
    
    return df_final


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    # 创建输出目录
    OUTPUT_DIR = "data/Processed_ELA_Phase1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 假设你有办法获取所有问题名称列表，或者遍历你的 ProblemName 枚举
    # 这里通过扫描实际存在的文件夹/文件后缀来动态获取问题名，或者直接用枚举
    problems = [p.value for p in ProblemName] if hasattr(ProblemName, '__iter__') else ["P1", "P2"] # 根据你本地的结构调整
    
    # 这里为了演示，假设直接从枚举中读取所有问题
    try:
        problems_list = [name for name, member in ProblemName.__members__.items()]
    except:
        # 如果获取不到，退化为写死的问题列表或自动扫描，请根据需要修改
        problems_list = ["F1", "F2", "F3"] # 替换为你的问题列表
        
    for prob in problems_list:
        df_processed = process_and_clean_ela(prob, target_n_coef=1000, target_block_coef=0.0)
        
        if df_processed is not None and not df_processed.empty:
            save_path = f"{OUTPUT_DIR}/cleaned_ela_{prob}.csv"
            df_processed.to_csv(save_path)
            print(f"  [+] Saved cleaned data to {save_path}")

    print("\n✅ Phase 1: Feature processing and collinearity cleaning completed.")