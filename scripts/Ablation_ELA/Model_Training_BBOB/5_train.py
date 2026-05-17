# fmt: off
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
# fmt; on


ELA_PATH = "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv"
PERF_PATH = "data/Ablation_ELA/algorithm_auc_performance.csv"
MODEL_SAVE_PATH = "models/sota_as_models.joblib"


def train_algorithm_selection_models():
    if not os.path.exists(ELA_PATH) or not os.path.exists(PERF_PATH):
        print("[!]Error: The feature file or performance file cannot be found. ")
        return

    ela_df = pd.read_csv(ELA_PATH)
    perf_df = pd.read_csv(PERF_PATH)
    bbob_ela = ela_df[ela_df['fid'].between(1, 24)].copy()
    print("Identifying the optimal algorithm for each BBOB instance")
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
    # X_clf = train_df_clf[feature_cols].fillna(
    #     train_df_clf[feature_cols].median())
    X_clf = train_df_clf[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_clf = X_clf.fillna(X_clf.median()).fillna(0)
    y_clf = train_df_clf['target_best_alg']
    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf)
    print("\nTraining a classification model (algorithm selector)...")
    clf = RandomForestClassifier(
        n_estimators=200, max_features='log2', n_jobs=-1, random_state=42)
    logo = LeaveOneGroupOut()
    groups_clf = train_df_clf['fid']

    y_pred_clf = cross_val_predict(
        clf, X_clf, y_clf_encoded, groups=groups_clf, cv=logo)
    print(
        f"Accuracy of the LOFO classification model: {accuracy_score(y_clf_encoded, y_pred_clf):.4f}")
    clf.fit(X_clf, y_clf_encoded)
    print("\nTraining a regression model (performance predictor)...")

    X_reg_raw = train_df_reg[feature_cols + ['algname']]
    X_reg_encoded = pd.get_dummies(X_reg_raw, columns=['algname'])

    X_reg = X_reg_encoded.replace([np.inf, -np.inf], np.nan)

    X_reg = X_reg.fillna(X_reg.median()).fillna(0)

    y_reg = train_df_reg['auc_mean'].replace([np.inf, -np.inf], np.nan)
    valid_idx = y_reg.notna()
    X_reg = X_reg[valid_idx]
    y_reg = y_reg[valid_idx]
    groups_reg = train_df_reg['fid'][valid_idx]

    reg = RandomForestRegressor(
        n_estimators=200, max_features='sqrt', n_jobs=-1, random_state=42)

    try:
        y_pred_reg = cross_val_predict(
            reg, X_reg, y_reg, groups=groups_reg, cv=logo)
        print(
            f"Regression model LOFO R2 score: {r2_score(y_reg, y_pred_reg):.4f}")
        print(
            f"Regression model LOFO MAE: {mean_absolute_error(y_reg, y_pred_reg):.4f}")
    except ValueError as e:
        print(f"Error: {e}")
        is_finite = np.all(np.isfinite(X_reg), axis=0)
        bad_cols = X_reg.columns[~is_finite].tolist()
        print(f"Error col: {bad_cols}")

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
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_algorithm_selection_models()
