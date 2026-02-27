import os
import numpy as np
import pandas as pd
import xgboost as xgb


class SeparabilityReport:
    separable: bool
    percent_noncompliance: float
    mean_norm_interaction: float
    max_norm_interaction: float
    per_pair_score: dict  # {(i,j): score}
    details: dict         # misc diagnostics


def evaluate_separability(f, dim, bounds=(-5.0, 5.0), samples=128, h=None,
                          tol=1e-6, rng=None):
    """
    Black-box separability test for f: R^dim -> R.
    Uses finite-difference Hessian cross-terms + a superposition check.

    Parameters
    ----------
    f : callable
        Function taking 1D np.ndarray (len=dim) -> float.
    dim : int
        Dimensionality.
    bounds : tuple(float, float)
        Box to sample from (same for all dims).
    samples : int
        Number of random points for finite-difference probing.
    h : float or None
        Step size for finite differences. If None, set to 1e-3 * (ub-lb).
    tol : float
        Threshold on normalized interaction to count as non-compliant.
    rng : np.random.Generator or None
        For reproducibility.

    Returns
    -------
    SeparabilityReport
    """
    if rng is None:
        rng = np.random.default_rng()
    lb, ub = map(float, bounds)
    box = ub - lb
    if h is None:
        h = 1e-3 * box
    eps = 1e-12

    # Draw sample points inside the box
    X = rng.uniform(lb + 2*h, ub - 2*h, size=(samples, dim))

    # Helpers for finite differences
    def e(i):
        v = np.zeros(dim)
        v[i] = 1.0
        return v

    def f_at(x):
        return float(f(x))

    def d2_ij(x, i, j):
        """Central difference mixed second derivative ∂^2 f / ∂x_i ∂x_j."""
        ei, ej = e(i), e(j)
        return (
            f_at(x + h*ei + h*ej) - f_at(x + h*ei - h*ej)
            - f_at(x - h*ei + h*ej) + f_at(x - h*ei - h*ej)
        ) / (4*h*h)

    def d2_ii(x, i):
        """Central difference second derivative ∂^2 f / ∂x_i^2."""
        ei = e(i)
        return (f_at(x + h*ei) - 2.0*f_at(x) + f_at(x - h*ei)) / (h*h)

    # Estimate diagonal curvature scales per dim for normalization
    Hii_vals = np.zeros((samples, dim))
    for s in range(samples):
        x = X[s]
        fx = f_at(x)  # cache center eval to reduce calls on diagonal calc
        for i in range(dim):
            ei = np.zeros(dim)
            ei[i] = 1.0
            Hii_vals[s, i] = (f_at(x + h*ei) - 2.0*fx + f_at(x - h*ei)) / (h*h)

    # Robust scale per axis: median absolute curvature (avoid zero division)
    scale_i = np.median(np.abs(Hii_vals), axis=0) + eps

    # Compute normalized cross interactions
    # aggregated per pair over samples (median of |H_ij|/sqrt(scale_i*scale_j))
    pair_scores = {}
    all_norm_ijs = []  # for overall stats
    per_pair_all = {(i, j): [] for i in range(dim) for j in range(i+1, dim)}

    for s in range(samples):
        x = X[s]
        # Precompute f(x) once for superposition later
        for i in range(dim):
            pass  # nothing extra here; left for symmetry

        for i in range(dim):
            for j in range(i+1, dim):
                Hij = d2_ij(x, i, j)
                norm = abs(Hij) / np.sqrt(scale_i[i]*scale_i[j])
                per_pair_all[(i, j)].append(norm)
                all_norm_ijs.append(norm)

    for (i, j), arr in per_pair_all.items():
        pair_scores[(i, j)] = float(np.median(arr))

    mean_norm_interaction = float(
        np.mean(all_norm_ijs)) if all_norm_ijs else 0.0
    max_norm_interaction = float(np.max(all_norm_ijs)) if all_norm_ijs else 0.0

    # Superposition sanity check: does Δ_i depend on j?
    # If separable, Δ_i(x) ≈ f(x+he_i)-f(x) should not change when j is perturbed.
    sup_violations = 0
    sup_total = 0
    for s in range(samples):
        x = X[s]
        fx = f_at(x)
        for i in range(dim):
            ei = e(i)
            d_i = f_at(x + h*ei) - fx
            for j in range(dim):
                if j == i:
                    continue
                ej = e(j)
                d_i_with_j = f_at(x + h*ej + h*ei) - f_at(x + h*ej)
                # Relative interaction score (bounded, scale-free-ish)
                denom = abs(d_i) + abs(d_i_with_j) + eps
                rel = abs(d_i_with_j - d_i) / denom
                sup_total += 1
                if rel > tol:
                    sup_violations += 1

    # Combine both notions into a single % non-compliance.
    # 50/50 weight between Hessian cross-terms and superposition violations.
    # For Hessian: count fraction above tol
    hess_violations = np.sum(np.array(all_norm_ijs) > tol)
    hess_total = len(all_norm_ijs)
    frac_hess = (hess_violations / max(1, hess_total))
    frac_sup = (sup_violations / max(1, sup_total))
    percent_noncompliance = 100.0 * (0.5*frac_hess + 0.5*frac_sup)

    separable = percent_noncompliance < 0.5  # basically “all but numerical noise”

    details = {
        "tol": tol,
        "h": h,
        "samples": samples,
        "bounds": (lb, ub),
        "diag_curvature_scale_per_dim": scale_i.tolist(),
        "hessian_fraction_violations": frac_hess,
        "superposition_fraction_violations": frac_sup,
    }

    return SeparabilityReport(
        separable=separable,
        percent_noncompliance=float(percent_noncompliance),
        mean_norm_interaction=mean_norm_interaction,
        max_norm_interaction=max_norm_interaction,
        per_pair_score={str(k): v for k, v in pair_scores.items()},
        details=details,
    )


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



if not os.path.exists("/data/hyin/GP_exp/data/description/"):
    os.makedirs("/data/hyin/GP_exp/data/description/")
problem_names = [
    "meta_surface",
    # "meta_surface_solver",
    "photonic_10layers_bragg",
    "photonic_20layers_bragg",
    "photonic_2layers_ellipsometry",
    "photonic_10layers_photovoltaic",
]
feature_descriptions = {
    # "Basins": "Basin size homogeneity, meaning the size relation (largest to smallest) of all basins of attraction should be homogeneous.",
    # "Separable": "Separable, meaning independent functions per dimension. Meaning, a problem may be partitioned into subproblems which are then of lower dimensionality and should be considerably easier to solve.",
    "GlobalLocal": "It should have a global local minima contrast, GlobalLocal refers to the difference between global and local peaks in comparison to the average fitness level of a problem. It thus determines if very good peaks are easily recognized as such.",
    "Multimodality": "it should be multimodal, Multimodality refers to the number of local minima of a problem.",
    "Structure": "It should have a clear global structure. Global structure is what remains after deleting all non-optimal points.",
    # "Homogeneous": "The search space should be homogeneous. Which refers to a search space without phase transitions. Its overall appearance is similar in different search space areas.",
    "NOT Homogeneous": "The search space should be not homogeneous. Which refers to a search space with phase transitions. Its overall appearance is different in different search space areas.",
    "NOT Basins": "The search space should be not have basin size homogeneity. Which refers to a search space where the size relation (largest to smallest) of all basins of attraction is not homogeneous.",
}
all_features = list(feature_descriptions.keys())

description_prefix = """
Here are the descriptions of several high-level landscape features:
"""
for feature in all_features:
    description_prefix += f"- {feature}: {feature_descriptions[feature]}\n"
for problem_name in problem_names:
    all_features_pandas = load_real_problem_ela(
        f"/data/hyin/GP_exp/data/ELA/ela_{problem_name}/ela_60.csv")
    all_features_mean = all_features_pandas.mean()
    feature_results = {}
    for feature in all_features:
        # if feature == "Separable":
        #     report = evaluate_separability(problem, DIM, bounds=bounds, samples=1024)
        #     feature_results[f"{feature} - {DIM}D"] = 1 - (report.percent_noncompliance / 100.0)
        inverse = False
        if feature in ["NOT Homogeneous", "NOT Basins"]:
            feature_key = feature.replace("NOT ", "")
            inverse = True
        else:
            feature_key = feature
        model = xgb.XGBClassifier(objective="binary:logistic")
        model.load_model(
            f"/data/hyin/GP_exp/data/dimensions/model_Groups_{feature_key}_scaled_new.json")
        input_df = pd.DataFrame([all_features_mean],
                                columns=all_features_pandas.columns)
        if inverse:
            feature_results[f"{feature}"] = 1 - model.predict_proba(input_df)[0][1]
        else:
            feature_results[f"{feature}"] = model.predict_proba(input_df)[0][1]
    description = description_prefix + "The problem that the algorithm is going to deal with has"
    for feature in all_features:
        description += f" {feature_results[feature]*100:.3f}% {feature},"
    description = description[:-1] + ". Please design an algorithm that can efficiently solve such problem.\n"
    with open(f"/data/hyin/GP_exp/data/description/{problem_name}.txt", "w") as f:
        f.write(description)
