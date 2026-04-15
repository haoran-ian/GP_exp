import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 环境准备 ---
bundle = joblib.load("data/Ablation_ELA/models/sota_as_models.joblib")
reg = bundle['regressor']
feature_cols = bundle['feature_cols']
reg_feature_cols = bundle['reg_feature_cols']

ela_df = pd.read_csv("data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv")
my_problems = ela_df[ela_df['fid'] < 1].copy()

def get_feature_group(feature_name):
    if 'ela_meta' in feature_name: return 'Meta-model'
    if 'ela_distr' in feature_name: return 'Distribution'
    if 'ela_level' in feature_name: return 'Level-set'
    if 'nbc' in feature_name: return 'Nearest Better'
    if 'ic' in feature_name: return 'Info. Content'
    if 'disp' in feature_name: return 'Dispersion'
    if 'pca' in feature_name: return 'PCA'
    return 'Others'

def run_visual_ablation(problem_name):
    print(f"📊 正在执行问题 [{problem_name}] 的消融可视化分析...")
    prob_data = my_problems[my_problems['problem_name'] == problem_name].copy()
    
    if prob_data.empty: return

    # --- 修复代码开始 ---
    # 获取模型训练时需要的特征和当前数据中实际存在的特征的交集
    available_features = [f for f in feature_cols if f in prob_data.columns]
    missing_features = set(feature_cols) - set(available_features)
    
    if missing_features:
        print(f"⚠️ 警告: 模型需要的 {len(missing_features)} 个特征在当前数据中缺失 (例如: {list(missing_features)[:3]})。将补 0。")

    # 提取存在的特征并清洗
    X_base_raw = prob_data[available_features].replace([np.inf, -np.inf], np.nan)
    X_base_raw = X_base_raw.fillna(X_base_raw.median()).fillna(0)
    
    # 【关键步骤】对于模型需要但数据中缺失的特征，强制补 0，确保维度对齐
    for m_feat in missing_features:
        X_base_raw[m_feat] = 0.0
        
    # 重新排序，确保特征顺序与训练时完全一致
    X_base_raw = X_base_raw[feature_cols]
    # --- 修复代码结束 ---
    
    # ... 剩下的代码 (algnames 提取, n_repeats 循环等)
    
    # 提取所有涉及的算法名
    algnames = [c.replace('algname_', '') for c in reg_feature_cols if c.startswith('algname_')]
    
    # 记录每个特征的 Impact
    results = []

    # 为了增加稳健性，每个特征置换 5 次取平均
    n_repeats = 5

    for feat in feature_cols:
        print 
        impacts = []
        for _ in range(n_repeats):
            for alg in algnames:
                # 构造基准输入
                X_input = X_base_raw.copy()
                for a in algnames:
                    X_input[f'algname_{a}'] = 1 if a == alg else 0
                
                base_pred = reg.predict(X_input).mean()
                
                # 执行置换 (Permutation)
                X_ablated = X_input.copy()
                X_ablated[feat] = np.random.permutation(X_ablated[feat].values)
                
                ablated_pred = reg.predict(X_ablated).mean()
                # Impact = 性能损失 (AUC增加)
                impacts.append(ablated_pred - base_pred)
        
        results.append({
            'Feature': feat,
            'Group': get_feature_group(feat),
            'Impact': np.mean(impacts)
        })

    df_res = pd.DataFrame(results)

    # --- 可视化 1: Top 15 关键特征 ---
    plt.figure(figsize=(10, 8))
    top_15 = df_res.sort_values(by='Impact', ascending=False).head(15)
    sns.barplot(data=top_15, x='Impact', y='Feature', hue='Group', dodge=False)
    plt.title(f"Top 15 Most Critical ELA Features for {problem_name}\n(Impact on Predicted AUC)")
    plt.xlabel("Performance Loss (Higher = More Important)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/ablation_top15_{problem_name}.png")

    # --- 可视化 2: ELA 特征组重要性占比 (TreeMap 或 Boxplot) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_res, x='Group', y='Impact', palette='Set3')
    plt.title(f"Impact Distribution by ELA Feature Groups ({problem_name})")
    plt.ylabel("Performance Loss")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/ablation_groups_{problem_name}.png")

# 运行分析
os.makedirs("results", exist_ok=True)
for p in my_problems['problem_name'].unique():
    run_visual_ablation(p)