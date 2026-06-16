# fmt; off
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# fmt: on

MODEL_PATH = "data/Combined/models/bbob_mabbob_llm_mixed_auc_source_normalized_regressor_as_model.joblib"
bundle = joblib.load(MODEL_PATH)
reg = bundle['regressor']
feature_cols = bundle['feature_cols']
reg_feature_cols = bundle['reg_feature_cols']


def force_single_thread_model(model):
    """Force sklearn estimators/pipelines to use one joblib worker."""
    if hasattr(model, "get_params") and hasattr(model, "set_params"):
        params = model.get_params()
        if "n_jobs" in params:
            model.set_params(n_jobs=1)

        # This also handles nested estimators such as Pipeline steps:
        # e.g. regressor__n_jobs, model__n_jobs, etc.
        nested_n_jobs = {
            key: 1 for key in params
            if key.endswith("__n_jobs")
        }
        if nested_n_jobs:
            model.set_params(**nested_n_jobs)

    return model


reg = force_single_thread_model(reg)

ela_df = pd.read_csv(
    "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv")
my_problems = ela_df[ela_df['fid'] < 1].copy()


def get_feature_group(feature_name):
    if 'ela_meta' in feature_name:
        return 'Meta-model'
    if 'ela_distr' in feature_name:
        return 'Distribution'
    if 'ela_level' in feature_name:
        return 'Level-set'
    if 'nbc' in feature_name:
        return 'Nearest Better'
    if 'ic' in feature_name:
        return 'Info. Content'
    if 'disp' in feature_name:
        return 'Dispersion'
    if 'pca' in feature_name:
        return 'PCA'
    return 'Others'


def run_visual_ablation(problem_name):
    print(f"Running ablation study on [{problem_name}]...")
    prob_data = my_problems[my_problems['problem_name'] == problem_name].copy()

    if prob_data.empty:
        return

    available_features = [f for f in feature_cols if f in prob_data.columns]
    missing_features = set(feature_cols) - set(available_features)

    if missing_features:
        print(
            f"Warning: {len(missing_features)} features missing, will be filled as 0.")

    X_base_raw = prob_data[available_features].replace(
        [np.inf, -np.inf], np.nan)
    X_base_raw = X_base_raw.fillna(X_base_raw.median()).fillna(0)

    for m_feat in missing_features:
        X_base_raw[m_feat] = 0.0

    X_base_raw = X_base_raw[feature_cols]
    algnames = [c.replace('algname_', '')
                for c in reg_feature_cols if c.startswith('algname_')]

    results = []

    n_repeats = 5

    for feat in feature_cols:
        impacts = []
        for _ in range(n_repeats):
            for alg in algnames:
                X_input = X_base_raw.copy()
                for a in algnames:
                    X_input[f'algname_{a}'] = 1 if a == alg else 0

                base_pred = reg.predict(X_input).mean()

                X_ablated = X_input.copy()
                X_ablated[feat] = np.random.permutation(X_ablated[feat].values)

                ablated_pred = reg.predict(X_ablated).mean()
                impacts.append(ablated_pred - base_pred)

        results.append({
            'Feature': feat,
            'Group': get_feature_group(feat),
            'Impact': np.mean(impacts)
        })

    df_res = pd.DataFrame(results)

    plt.figure(figsize=(10, 8))
    top_15 = df_res.sort_values(by='Impact', ascending=False).head(15)
    sns.barplot(data=top_15, x='Impact', y='Feature', hue='Group', dodge=False)
    plt.title(
        f"Top 15 Most Critical ELA Features for {problem_name}\n(Impact on Predicted AUC)")
    plt.xlabel("Performance Loss (Higher = More Important)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"results/Combined/ablation_top15_{problem_name}.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_res, x='Group', y='Impact', palette='Set3')
    plt.title(f"Impact Distribution by ELA Feature Groups ({problem_name})")
    plt.ylabel("Performance Loss")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"results/Combined/ablation_groups_{problem_name}.png")


os.makedirs("results", exist_ok=True)
for p in my_problems['problem_name'].unique():
    run_visual_ablation(p)
