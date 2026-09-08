
# -*- coding: utf-8 -*-
"""
Visualize quality of source-specific MLP regressors.

Assumes you have run:
    6_train_source_specific_mlp_and_realworld_feature_set_ablation.py

Expected input directory:
    data/Combined/source_specific_mlp_regressors/

Expected files:
    BBOB_MLP_cv_predictions.csv
    MABBOB_MLP_cv_predictions.csv
    LLM_MLP_cv_predictions.csv
    source_specific_model_training_summary.csv

Outputs:
    data/Combined/source_specific_mlp_regressors/mlp_quality_plots/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MLP_DIR = "data/Combined/source_specific_mlp_regressors"
OUT_DIR = os.path.join(MLP_DIR, "mlp_quality_plots")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAMES = ["BBOB_MLP", "MABBOB_MLP", "LLM_MLP"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return np.nan
    return float(r2_score(y_true, y_pred))


def safe_spearman(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def parse_source_from_model_name(model_name):
    if model_name.startswith("BBOB"):
        return "BBOB"
    if model_name.startswith("MABBOB"):
        return "MABBOB"
    if model_name.startswith("LLM"):
        return "LLM"
    return "UNKNOWN"


def require_cols(df, cols, path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def load_prediction_files():
    parts = []

    for model_name in MODEL_NAMES:
        path = os.path.join(MLP_DIR, f"{model_name}_cv_predictions.csv")
        if not os.path.exists(path):
            print(f"[Skip] Missing prediction file: {path}")
            continue

        df = pd.read_csv(path)
        require_cols(
            df,
            ["target_auc", "pred_target_auc", "problem_key", "algname"],
            path,
        )

        df["model_name"] = model_name
        df["training_source"] = parse_source_from_model_name(model_name)
        df["error"] = df["pred_target_auc"] - df["target_auc"]
        df["abs_error"] = np.abs(df["error"])

        if "problem_type" not in df.columns:
            df["problem_type"] = df["training_source"]

        parts.append(df)

    if not parts:
        raise RuntimeError(
            f"No MLP CV prediction files found in {MLP_DIR}. "
            "Run the MLP training script first."
        )

    return pd.concat(parts, ignore_index=True)


def compute_metrics(pred_df):
    rows = []

    for model_name, sub in pred_df.groupby("model_name"):
        y = sub["target_auc"].to_numpy(float)
        p = sub["pred_target_auc"].to_numpy(float)

        rows.append({
            "model_name": model_name,
            "training_source": sub["training_source"].iloc[0],
            "n_rows": len(sub),
            "n_problems": sub["problem_key"].nunique(),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": rmse(y, p),
            "r2": safe_r2(y, p),
            "spearman": safe_spearman(y, p),
            "median_abs_error": float(np.median(np.abs(p - y))),
            "p90_abs_error": float(np.quantile(np.abs(p - y), 0.90)),
            "p95_abs_error": float(np.quantile(np.abs(p - y), 0.95)),
        })

    return pd.DataFrame(rows)


def plot_pred_vs_true_single(sub, model_name):
    y = sub["target_auc"].to_numpy(float)
    p = sub["pred_target_auc"].to_numpy(float)

    mae = mean_absolute_error(y, p)
    sp = safe_spearman(y, p)
    r2 = safe_r2(y, p)

    lo = float(np.nanmin([y.min(), p.min()]))
    hi = float(np.nanmax([y.max(), p.max()]))
    pad = 0.05 * (hi - lo + 1e-12)

    plt.figure(figsize=(6.5, 6))
    plt.scatter(y, p, s=18, alpha=0.55)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    plt.xlabel("True per-problem-normalized target")
    plt.ylabel("Predicted target")
    plt.title(f"{model_name}: predicted vs true")
    plt.grid(alpha=0.25)

    text = f"MAE = {mae:.4f}\nR² = {r2:.4f}\nSpearman = {sp:.4f}\nn = {len(sub)}"
    plt.text(
        0.04,
        0.96,
        text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{model_name}_pred_vs_true.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_residual_single(sub, model_name):
    pred = sub["pred_target_auc"].to_numpy(float)
    err = sub["error"].to_numpy(float)

    plt.figure(figsize=(7, 5))
    plt.scatter(pred, err, s=18, alpha=0.55)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted target")
    plt.ylabel("Prediction error: pred - true")
    plt.title(f"{model_name}: residual plot")
    plt.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{model_name}_residual_plot.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_abs_error_distribution_single(sub, model_name):
    abs_err = sub["abs_error"].to_numpy(float)

    plt.figure(figsize=(7, 5))
    plt.hist(abs_err, bins=40, alpha=0.8)
    plt.axvline(np.mean(abs_err), linestyle="--", linewidth=1, label=f"mean={np.mean(abs_err):.4f}")
    plt.axvline(np.median(abs_err), linestyle=":", linewidth=1, label=f"median={np.median(abs_err):.4f}")
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.title(f"{model_name}: absolute error distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{model_name}_abs_error_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_per_problem_error(sub, model_name):
    per_problem = (
        sub.groupby("problem_key", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            n_rows=("abs_error", "size"),
        )
        .sort_values("mae", ascending=False)
    )

    per_problem.to_csv(
        os.path.join(OUT_DIR, f"{model_name}_per_problem_error.csv"),
        index=False,
    )

    plot_df = per_problem.head(30).sort_values("mae", ascending=True)

    plt.figure(figsize=(9, max(5, 0.30 * len(plot_df))))
    plt.barh(plot_df["problem_key"], plot_df["mae"])
    plt.xlabel("Mean absolute error")
    plt.ylabel("Problem")
    plt.title(f"{model_name}: per-problem MAE, top 30 worst")
    plt.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{model_name}_per_problem_mae_top30.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_combined_pred_vs_true(pred_df):
    plt.figure(figsize=(7.2, 6.2))

    for model_name, sub in pred_df.groupby("model_name"):
        plt.scatter(
            sub["target_auc"],
            sub["pred_target_auc"],
            s=18,
            alpha=0.50,
            label=f"{model_name} (n={len(sub)})",
        )

    y = pred_df["target_auc"].to_numpy(float)
    p = pred_df["pred_target_auc"].to_numpy(float)
    lo = float(np.nanmin([y.min(), p.min()]))
    hi = float(np.nanmax([y.max(), p.max()]))
    pad = 0.05 * (hi - lo + 1e-12)

    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
    plt.xlabel("True per-problem-normalized target")
    plt.ylabel("Predicted target")
    plt.title("MLP regressors: predicted vs true")
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "all_mlp_pred_vs_true.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_metric_bar(metrics_df, metric, title, ylabel, higher_is_better=False):
    df = metrics_df.sort_values(metric, ascending=not higher_is_better).copy()

    plt.figure(figsize=(8, 5))
    plt.bar(df["model_name"], df[metric])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)

    for i, row in enumerate(df.itertuples()):
        val = getattr(row, metric)
        if np.isfinite(val):
            plt.text(i, val, f"{val:.4g}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"mlp_quality_{metric}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_error_boxplot(pred_df):
    model_names = list(pred_df["model_name"].drop_duplicates())
    data = [
        pred_df[pred_df["model_name"] == m]["abs_error"].to_numpy(float)
        for m in model_names
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=model_names, showfliers=False)
    plt.ylabel("Absolute error")
    plt.title("MLP regressors: absolute error distribution by source")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "all_mlp_abs_error_boxplot.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_algorithm_error_heatmap(pred_df):
    pivot = (
        pred_df.groupby(["model_name", "algname"])["abs_error"]
        .mean()
        .reset_index()
        .pivot(index="model_name", columns="algname", values="abs_error")
    )

    if pivot.shape[1] > 40:
        top_algs = pivot.mean(axis=0).sort_values(ascending=False).head(40).index
        pivot = pivot[top_algs]

    fig_width = max(10, 0.35 * pivot.shape[1])
    fig_height = max(3.5, 0.6 * pivot.shape[0])

    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(pivot.to_numpy(float), aspect="auto")
    plt.colorbar(im, label="Mean absolute error")

    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title("Mean absolute error by MLP model and algorithm")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "all_mlp_algorithm_error_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

    pivot.to_csv(os.path.join(OUT_DIR, "all_mlp_algorithm_error_matrix.csv"))


def main():
    print("Loading MLP prediction files...")
    pred_df = load_prediction_files()

    pred_path = os.path.join(OUT_DIR, "all_mlp_cv_predictions_with_errors.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved: {pred_path}")

    metrics_df = compute_metrics(pred_df)
    metrics_path = os.path.join(OUT_DIR, "mlp_quality_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    print("\n=== MLP quality metrics ===")
    print(metrics_df)

    for model_name, sub in pred_df.groupby("model_name"):
        plot_pred_vs_true_single(sub, model_name)
        plot_residual_single(sub, model_name)
        plot_abs_error_distribution_single(sub, model_name)
        plot_per_problem_error(sub, model_name)

    plot_combined_pred_vs_true(pred_df)
    plot_error_boxplot(pred_df)

    plot_metric_bar(metrics_df, "mae", "MLP regressors: CV MAE", "MAE", higher_is_better=False)
    plot_metric_bar(metrics_df, "rmse", "MLP regressors: CV RMSE", "RMSE", higher_is_better=False)
    plot_metric_bar(metrics_df, "spearman", "MLP regressors: CV Spearman", "Spearman correlation", higher_is_better=True)
    plot_metric_bar(metrics_df, "r2", "MLP regressors: CV R²", "R²", higher_is_better=True)

    if "algname" in pred_df.columns:
        plot_algorithm_error_heatmap(pred_df)

    config = {
        "input_dir": MLP_DIR,
        "output_dir": OUT_DIR,
        "model_names": MODEL_NAMES,
        "target": "per-problem-normalized target_auc",
        "prediction_column": "pred_target_auc",
        "true_column": "target_auc",
    }
    with open(os.path.join(OUT_DIR, "mlp_quality_plot_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")
    print(f"Quality plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
