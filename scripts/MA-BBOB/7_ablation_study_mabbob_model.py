import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = "data/MABBOB/models/mabbob_uniform_as_models.joblib"
ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"

OUT_DIR = "results_mabbob_ablation"
os.makedirs(OUT_DIR, exist_ok=True)


def get_feature_group(feature_name):
    if "ela_meta" in feature_name:
        return "Meta-model"
    if "ela_distr" in feature_name or "ela_distribution" in feature_name:
        return "Distribution"
    if "ela_level" in feature_name:
        return "Level-set"
    if "nbc" in feature_name:
        return "Nearest Better"
    if "ic" in feature_name or "information_content" in feature_name:
        return "Info. Content"
    if "disp" in feature_name or "dispersion" in feature_name:
        return "Dispersion"
    if "pca" in feature_name:
        return "PCA"
    if "limo" in feature_name:
        return "Linear Model"
    if "cm_" in feature_name:
        return "Cell Mapping"
    if "gradient" in feature_name:
        return "Gradient"
    if "hill_climbing" in feature_name:
        return "Hill Climbing"
    if "length_scale" in feature_name:
        return "Length Scale"
    if "fla_metrics" in feature_name:
        return "Sobol / FLA"
    return "Others"


def prepare_real_world_X(prob_data, feature_cols):
    available_features = [f for f in feature_cols if f in prob_data.columns]
    missing_features = sorted(set(feature_cols) - set(available_features))

    X = prob_data[available_features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    for f in missing_features:
        X[f] = 0.0

    X = X[feature_cols]

    if missing_features:
        print(f"Filled {len(missing_features)} missing features with 0.")

    return X


def run_visual_ablation(problem_name, prob_data, reg, feature_cols, reg_feature_cols):
    print(f"Running MA-BBOB-trained ablation on [{problem_name}]...")

    if prob_data.empty:
        return None

    X_base_raw = prepare_real_world_X(prob_data, feature_cols)

    algnames = [
        c.replace("algname_", "")
        for c in reg_feature_cols
        if c.startswith("algname_")
    ]

    results = []
    n_repeats = 10

    for feat in feature_cols:
        impacts = []

        for _ in range(n_repeats):
            for alg in algnames:
                X_input = X_base_raw.copy()

                # 补齐 algorithm one-hot columns
                for a in algnames:
                    X_input[f"algname_{a}"] = 1 if a == alg else 0

                # 补齐 regressor 训练时有、当前没有的列
                for c in reg_feature_cols:
                    if c not in X_input.columns:
                        X_input[c] = 0.0

                X_input = X_input[reg_feature_cols]

                base_pred = reg.predict(X_input).mean()

                X_ablated = X_input.copy()
                if feat in X_ablated.columns:
                    X_ablated[feat] = np.random.permutation(X_ablated[feat].values)

                ablated_pred = reg.predict(X_ablated).mean()
                impacts.append(ablated_pred - base_pred)

        results.append({
            "problem_name": problem_name,
            "Feature": feat,
            "Group": get_feature_group(feat),
            "Impact": float(np.mean(impacts)),
        })

    df_res = pd.DataFrame(results)

    csv_path = os.path.join(OUT_DIR, f"ablation_values_{problem_name}.csv")
    df_res.to_csv(csv_path, index=False)

    # Figure 1: top 15
    top_15 = df_res.sort_values(by="Impact", ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_15))
    plt.barh(y_pos, top_15["Impact"].values)
    plt.yticks(y_pos, top_15["Feature"].values)
    plt.gca().invert_yaxis()
    plt.xlabel("Predicted AUC change after permutation")
    plt.title(f"Top 15 ELA Features for {problem_name}\nModel trained on ELA-uniform MA-BBOB")
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, f"mabbob_ablation_top15_{problem_name}.png")
    plt.savefig(fig_path, dpi=250)
    plt.close()

    # Figure 2: group means
    group_df = (
        df_res.groupby("Group")["Impact"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(group_df))
    plt.barh(y_pos, group_df["mean"].values)
    plt.yticks(y_pos, group_df["Group"].values)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean predicted AUC change after permutation")
    plt.title(f"ELA Feature Group Importance for {problem_name}\nModel trained on ELA-uniform MA-BBOB")
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, f"mabbob_ablation_groups_{problem_name}.png")
    plt.savefig(fig_path, dpi=250)
    plt.close()

    print(f"[Saved] {csv_path}")

    return df_res


def main():
    bundle = joblib.load(MODEL_PATH)

    reg = bundle["regressor"]
    feature_cols = bundle["feature_cols"]
    reg_feature_cols = bundle["reg_feature_cols"]

    ela_df = pd.read_csv(ELA_PATH)

    # 你的原脚本把 fid < 1 作为 real-world problems
    real_world = ela_df[ela_df["fid"] < 1].copy()

    all_results = []

    for p in real_world["problem_name"].unique():
        prob_data = real_world[real_world["problem_name"] == p].copy()
        res = run_visual_ablation(
            problem_name=p,
            prob_data=prob_data,
            reg=reg,
            feature_cols=feature_cols,
            reg_feature_cols=reg_feature_cols,
        )
        if res is not None:
            all_results.append(res)

    if all_results:
        all_df = pd.concat(all_results, axis=0, ignore_index=True)
        all_path = os.path.join(OUT_DIR, "mabbob_ablation_all_real_world.csv")
        all_df.to_csv(all_path, index=False)
        print(f"[Saved] {all_path}")

    print("Done.")


if __name__ == "__main__":
    main()