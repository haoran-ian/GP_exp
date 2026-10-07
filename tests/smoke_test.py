import pandas as pd
import xgboost as xgb


def load_real_problem_ela(path: str):
    def sort_key(col):
        for i, key in enumerate(ela_set):
            if key in col:
                return i
        return len(ela_set)
    ela_set = ["ela_meta", "ela_distr", "nbc", "disp", "pca", "ic"]
    df_ela = pd.read_csv(path)
    selected_cols = [
        col for col in df_ela.columns
        if any(key in col for key in ela_set)
    ]
    selected_cols_sorted = sorted(selected_cols, key=sort_key)
    df = df_ela[selected_cols_sorted]
    return df


all_features = ["Separable", "GlobalLocal",
                "Multimodality", "Basins", "Homogeneous"]
all_features_pandas = load_real_problem_ela(
    "/data/hyin/GP_exp/data/ELA/ela_meta_surface/ela_60.csv")
all_features_mean = all_features_pandas.mean()
feature_results = {}
for feature in all_features:
    model = xgb.XGBClassifier(objective="binary:logistic")
    model.load_model(f"dimensions/model_Groups_{feature}_scaled_new.json")
    input_df = pd.DataFrame([all_features_mean],
                            columns=all_features_pandas.columns)
    feature_results[f"{feature}"] = model.predict_proba(input_df)[0][1]
print(feature_results)
