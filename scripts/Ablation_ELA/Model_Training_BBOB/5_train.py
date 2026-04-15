import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
PERF_PATH = "results/algorithm_auc_performance.csv"
MODEL_SAVE_PATH = "models/sota_as_models.joblib"


def train_algorithm_selection_models():
    if not os.path.exists(ELA_PATH) or not os.path.exists(PERF_PATH):
        print("❌ 错误：找不到特征文件或性能文件，请确保之前的 Pipeline 已运行。")
        return

    ela_df = pd.read_csv(ELA_PATH)
    perf_df = pd.read_csv(PERF_PATH)
    bbob_ela = ela_df[ela_df['fid'].between(1, 24)].copy()
    print("🎯 正在识别每个 BBOB 实例的最优算法 (Minimize AUC)...")
    best_alg_per_instance = perf_df[perf_df['fid'].between(1, 24)].copy()
    idx = best_alg_per_instance.groupby(['fid', 'iid', 'dim'])[
        'auc_mean'].idxmin()
    best_alg_labels = best_alg_per_instance.loc[idx, [
        'fid', 'iid', 'dim', 'algname']]
    best_alg_labels = best_alg_labels.rename(
        columns={'algname': 'target_best_alg'})

    train_df_clf = pd.merge(bbob_ela, best_alg_labels,
                            on=['fid', 'iid', 'dim'])

    train_df_reg = pd.merge(bbob_ela, perf_df, on=['fid', 'iid', 'dim'])

    meta_cols = ['problem_name', 'fid', 'iid', 'dim', 'seed']
    feature_cols = [c for c in bbob_ela.columns if c not in meta_cols]
    X_clf = train_df_clf[feature_cols].fillna(
        train_df_clf[feature_cols].median())
    y_clf = train_df_clf['target_best_alg']
    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf)
    print("\n🚀 正在训练分类模型 (算法选择器)...")
    clf = RandomForestClassifier(
        n_estimators=200, max_features='log2', n_jobs=-1, random_state=42)
    logo = LeaveOneGroupOut()
    groups_clf = train_df_clf['fid']

    y_pred_clf = cross_val_predict(
        clf, X_clf, y_clf_encoded, groups=groups_clf, cv=logo)
    print(f"✅ 分类模型 LOFO 准确率: {accuracy_score(y_clf_encoded, y_pred_clf):.4f}")
    clf.fit(X_clf, y_clf_encoded)
    print("\n📈 正在训练回归模型 (性能预测器)...")
    X_reg_raw = train_df_reg[feature_cols + ['algname']]
    X_reg = pd.get_dummies(X_reg_raw, columns=['algname'])
    X_reg = X_reg.fillna(X_reg.median())
    y_reg = train_df_reg['auc_mean']

    reg = RandomForestRegressor(
        n_estimators=200, max_features='sqrt', n_jobs=-1, random_state=42)
    groups_reg = train_df_reg['fid']

    y_pred_reg = cross_val_predict(
        reg, X_reg, y_reg, groups=groups_reg, cv=logo)
    print(f"✅ 回归模型 LOFO R2 分数: {r2_score(y_reg, y_pred_reg):.4f}")
    print(f"✅ 回归模型 LOFO MAE: {mean_absolute_error(y_reg, y_pred_reg):.4f}")
    reg.fit(X_reg, y_reg)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model_bundle = {
        'classifier': clf,
        'regressor': reg,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'reg_feature_cols': X_reg.columns.tolist(),
        'training_meta': {
            'n_samples_clf': len(X_clf),
            'n_samples_reg': len(X_reg),
            'n_features': len(feature_cols)
        }
    }
    joblib.dump(model_bundle, MODEL_SAVE_PATH)
    print(f"\n💾 模型已保存至: {MODEL_SAVE_PATH}")
    print("✨ 你现在可以使用该模型对你的自定义问题进行 ELA 特征消融预测了。")


if __name__ == "__main__":
    train_algorithm_selection_models()
