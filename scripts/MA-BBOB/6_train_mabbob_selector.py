import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


MABBOB_ELA_PATH = "data/MABBOB/mabbob_selected_ela.csv"
MABBOB_PERF_PATH = "data/MABBOB/mabbob_algorithm_auc_performance.csv"

MODEL_SAVE_PATH = "data/MABBOB/models/mabbob_uniform_as_models.joblib"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


def clean_X(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


def train_mabbob_algorithm_selection_models():
    if not os.path.exists(MABBOB_ELA_PATH):
        raise FileNotFoundError(MABBOB_ELA_PATH)

    if not os.path.exists(MABBOB_PERF_PATH):
        raise FileNotFoundError(MABBOB_PERF_PATH)

    ela_df = pd.read_csv(MABBOB_ELA_PATH)
    perf_df = pd.read_csv(MABBOB_PERF_PATH)

    # 统一 key
    if "mabbob_instance_id" not in ela_df.columns:
        ela_df["mabbob_instance_id"] = ela_df["instance_id"].astype(int)

    ela_df["fid"] = -100
    ela_df["iid"] = ela_df["mabbob_instance_id"].astype(int)
    ela_df["problem_name"] = ela_df["iid"].apply(lambda x: f"MABBOB_{int(x)}")

    perf_df["fid"] = -100
    perf_df["iid"] = perf_df["mabbob_instance_id"].astype(int)
    perf_df["problem_name"] = perf_df["iid"].apply(lambda x: f"MABBOB_{int(x)}")

    meta_cols = [
        "problem_type",
        "problem_name",
        "fid",
        "iid",
        "dim",
        "seed",
        "n_samples",
        "instance_id",
        "mabbob_instance_id",
        "selection_method",
    ]

    feature_cols = [
        c for c in ela_df.columns
        if c not in meta_cols
        and pd.api.types.is_numeric_dtype(ela_df[c])
        and not c.endswith(".FAILED")
    ]

    # 分类目标：每个 MA-BBOB instance 上 AUC 最小的 algorithm
    idx = perf_df.groupby(["fid", "iid", "dim"])["auc_mean"].idxmin()
    best_alg_labels = perf_df.loc[idx, ["fid", "iid", "dim", "algname"]]
    best_alg_labels = best_alg_labels.rename(columns={"algname": "target_best_alg"})

    train_df_clf = pd.merge(
        ela_df,
        best_alg_labels,
        on=["fid", "iid", "dim"],
        how="inner",
    )

    # 回归目标：每个 instance × algorithm 的 AUC
    train_df_reg = pd.merge(
        ela_df,
        perf_df,
        on=["fid", "iid", "dim"],
        how="inner",
        suffixes=("", "_perf"),
    )

    print(f"Training classifier rows: {len(train_df_clf)}")
    print(f"Training regressor rows: {len(train_df_reg)}")
    print(f"ELA features: {len(feature_cols)}")

    # ---------------------------
    # Classifier
    # ---------------------------
    X_clf = clean_X(train_df_clf[feature_cols])
    y_clf = train_df_clf["target_best_alg"]

    le = LabelEncoder()
    y_clf_encoded = le.fit_transform(y_clf)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="log2",
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    # MA-BBOB 没有 fid=1..24 这种天然 function group；
    # 这里用 instance_id 分组做 GroupKFold，避免同一个 instance 泄漏。
    groups_clf = train_df_clf["iid"].astype(int)

    n_groups = groups_clf.nunique()
    n_splits = min(5, n_groups)

    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        y_pred_clf = cross_val_predict(
            clf,
            X_clf,
            y_clf_encoded,
            groups=groups_clf,
            cv=gkf,
        )
        print(
            f"MA-BBOB GroupKFold classification accuracy: "
            f"{accuracy_score(y_clf_encoded, y_pred_clf):.4f}"
        )

    clf.fit(X_clf, y_clf_encoded)

    # ---------------------------
    # Regressor
    # ---------------------------
    X_reg_raw = train_df_reg[feature_cols + ["algname"]].copy()
    X_reg_encoded = pd.get_dummies(X_reg_raw, columns=["algname"])

    X_reg = clean_X(X_reg_encoded)
    y_reg = train_df_reg["auc_mean"].replace([np.inf, -np.inf], np.nan)

    valid_idx = y_reg.notna()
    X_reg = X_reg.loc[valid_idx]
    y_reg = y_reg.loc[valid_idx]
    groups_reg = train_df_reg.loc[valid_idx, "iid"].astype(int)

    reg = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    n_groups_reg = groups_reg.nunique()
    n_splits_reg = min(5, n_groups_reg)

    if n_splits_reg >= 2:
        gkf_reg = GroupKFold(n_splits=n_splits_reg)
        y_pred_reg = cross_val_predict(
            reg,
            X_reg,
            y_reg,
            groups=groups_reg,
            cv=gkf_reg,
        )
        print(f"MA-BBOB GroupKFold regression R2: {r2_score(y_reg, y_pred_reg):.4f}")
        print(f"MA-BBOB GroupKFold regression MAE: {mean_absolute_error(y_reg, y_pred_reg):.4f}")

    reg.fit(X_reg, y_reg)

    model_bundle = {
        "classifier": clf,
        "regressor": reg,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "reg_feature_cols": X_reg.columns.tolist(),
        "training_meta": {
            "train_source": "ELA-uniform MA-BBOB",
            "selection": "lhs_target_or_farthest",
            "n_samples_clf": len(X_clf),
            "n_samples_reg": len(X_reg),
            "n_features": len(feature_cols),
            "n_algorithms": train_df_reg["algname"].nunique(),
        },
    }

    joblib.dump(model_bundle, MODEL_SAVE_PATH)

    train_df_clf.to_csv("data/MABBOB/mabbob_training_table_clf.csv", index=False)
    train_df_reg.to_csv("data/MABBOB/mabbob_training_table_reg.csv", index=False)

    print(f"[Saved] {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_mabbob_algorithm_selection_models()