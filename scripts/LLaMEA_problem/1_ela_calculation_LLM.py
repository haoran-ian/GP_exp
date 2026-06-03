# fmt: off
import os
import sys
import warnings
import hashlib
import numpy as np
import pandas as pd
import pflacco.classical_ela_features as ela
sys.path.insert(0, os.getcwd())
from scipy.stats import qmc
from utils.extract_generated_problems import extract_llm_generated_problems
# fmt: on

warnings.filterwarnings("ignore")


# ============================================================
# 1. Global config
# ============================================================

OUT_DIR = "data/LLM"
OUT_FILE = os.path.join(OUT_DIR, "llm_generated_ela.csv")
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_DIM = 5
DEFAULT_LOWER_BOUND = -5.0
DEFAULT_UPPER_BOUND = 5.0
N_COEF = 1000             # n_samples = dim * N_COEF
RANDOM_SEED = 42


# ============================================================
# 2. Robust problem adapter
# ============================================================

def _to_array_bound(bound, dim, default):
    if bound is None:
        return np.full(dim, default, dtype=float)
    arr = np.asarray(bound, dtype=float)
    if arr.ndim == 0:
        return np.full(dim, float(arr), dtype=float)
    if len(arr) != dim:
        raise ValueError(f"Bound length {len(arr)} != dim {dim}")
    return arr.astype(float)


def get_problem_name(problem, idx):
    for attr in ["name", "problem_name", "__name__"]:
        if hasattr(problem, attr):
            value = getattr(problem, attr)
            if isinstance(value, str) and value:
                return value
    if isinstance(problem, dict):
        for key in ["name", "problem_name"]:
            if key in problem:
                return str(problem[key])
    return f"LLM_{idx:04d}"


def get_problem_dim(problem):
    # IOH-like
    if hasattr(problem, "meta_data") and hasattr(problem.meta_data, "n_variables"):
        return int(problem.meta_data.n_variables)

    # common attributes
    for attr in ["dim", "dimension", "n_variables", "n_var", "n_dims"]:
        if hasattr(problem, attr):
            return int(getattr(problem, attr))

    # dict-like
    if isinstance(problem, dict):
        for key in ["dim", "dimension", "n_variables", "n_var", "n_dims"]:
            if key in problem:
                return int(problem[key])
        if "bounds" in problem:
            bounds = problem["bounds"]
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lb = np.asarray(bounds[0])
                if lb.ndim > 0:
                    return len(lb)

    return int(DEFAULT_DIM)


def get_problem_bounds(problem, dim):
    # IOH-like
    if hasattr(problem, "bounds"):
        b = problem.bounds
        if hasattr(b, "lb") and hasattr(b, "ub"):
            return (
                _to_array_bound(b.lb, dim, DEFAULT_LOWER_BOUND),
                _to_array_bound(b.ub, dim, DEFAULT_UPPER_BOUND),
            )

    # common attributes
    lb = None
    ub = None
    for attr in ["lb", "lower_bound", "lower_bounds"]:
        if hasattr(problem, attr):
            lb = getattr(problem, attr)
            break
    for attr in ["ub", "upper_bound", "upper_bounds"]:
        if hasattr(problem, attr):
            ub = getattr(problem, attr)
            break

    # dict-like
    if isinstance(problem, dict):
        if "bounds" in problem:
            bounds = problem["bounds"]
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lb, ub = bounds
        lb = problem.get("lb", problem.get("lower_bound", problem.get("lower_bounds", lb)))
        ub = problem.get("ub", problem.get("upper_bound", problem.get("upper_bounds", ub)))

    return (
        _to_array_bound(lb, dim, DEFAULT_LOWER_BOUND),
        _to_array_bound(ub, dim, DEFAULT_UPPER_BOUND),
    )


def call_problem(problem, x):
    if isinstance(problem, dict):
        for key in ["func", "function", "objective", "fitness"]:
            if key in problem and callable(problem[key]):
                return problem[key](x)
        raise ValueError("Dict problem must contain a callable key: func/function/objective/fitness.")
    return problem(x)


# ============================================================
# 3. Sampling and feature calculation
# ============================================================

def stable_seed(base_seed, problem_name, idx):
    token = f"{problem_name}-{idx}-{base_seed}".encode("utf-8")
    h = hashlib.md5(token).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def sobol_sample(dim, n, seed, lower_bound, upper_bound):
    m = int(np.ceil(np.log2(n)))
    sampler = qmc.Sobol(d=dim, scramble=True, seed=int(seed))
    X01 = sampler.random_base2(m=m)[:n]
    return qmc.scale(X01, lower_bound, upper_bound)


def clean_y(y):
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(y)

    if not np.any(finite_mask):
        return np.zeros_like(y, dtype=float), y

    finite_y = y[finite_mask]
    y_clean = np.nan_to_num(
        y,
        nan=np.nanmedian(finite_y),
        posinf=np.max(finite_y),
        neginf=np.min(finite_y),
    )

    # Keep consistent with MA-BBOB pipeline: log1p(y - min(y)).
    y_shift = y_clean - np.min(y_clean)
    y_trans = np.log1p(y_shift)
    return y_trans, y_clean


def safe_add_features(feature_dict, prefix, func, *args, **kwargs):
    try:
        feats = func(*args, **kwargs)
        for k, v in feats.items():
            feature_dict[f"{prefix}.{k}"] = v
    except Exception as e:
        feature_dict[f"{prefix}.FAILED"] = 1
        feature_dict[f"{prefix}.ERROR"] = str(e)
    return feature_dict


def compute_ela_for_problem(problem, problem_id, problem_name):
    dim = get_problem_dim(problem)
    lb, ub = get_problem_bounds(problem, dim)

    n_samples = int(dim * N_COEF)
    seed = stable_seed(RANDOM_SEED, problem_name, problem_id)

    X = sobol_sample(dim, n_samples, seed, lb, ub)
    y_raw = np.array([call_problem(problem, x) for x in X], dtype=float)
    y_trans, y_clean = clean_y(y_raw)

    feats = {
        "problem_type": "LLM",
        "problem_name": problem_name,
        "fid": -200,
        "iid": int(problem_id),
        "llm_problem_id": int(problem_id),
        "dim": int(dim),
        "seed": int(seed),
        "n_samples": int(n_samples),
        "lower_bound_min": float(np.min(lb)),
        "lower_bound_max": float(np.max(lb)),
        "upper_bound_min": float(np.min(ub)),
        "upper_bound_max": float(np.max(ub)),
        "y_min": float(np.min(y_clean)),
        "y_max": float(np.max(y_clean)),
        "y_mean": float(np.mean(y_clean)),
        "y_std": float(np.std(y_clean)),
    }

    # Same core classical ELA sets as the MA-BBOB script.
    feats = safe_add_features(feats, "ela_meta", ela.calculate_ela_meta, X, y_trans)
    feats = safe_add_features(feats, "ela_distr", ela.calculate_ela_distribution, X, y_trans)
    feats = safe_add_features(feats, "ela_level", ela.calculate_ela_level, X, y_trans)
    feats = safe_add_features(feats, "pca", ela.calculate_pca, X, y_trans)
    feats = safe_add_features(
        feats,
        "limo",
        ela.calculate_limo,
        X,
        y_trans,
        lower_bound=lb,
        upper_bound=ub,
        force=True,
    )
    feats = safe_add_features(feats, "nbc", ela.calculate_nbc, X, y_trans, minimize=True)
    feats = safe_add_features(feats, "disp", ela.calculate_dispersion, X, y_trans, minimize=True)
    feats = safe_add_features(
        feats,
        "ic",
        ela.calculate_information_content,
        X,
        y_trans,
        seed=seed,
    )

    return feats


def main():
    problems = extract_llm_generated_problems()
    print(f"Loaded {len(problems)} LLM-generated problems.")

    rows = []
    for idx, problem in enumerate(problems):
        problem_id = idx + 1
        problem_name = get_problem_name(problem, problem_id)
        print(f"[ELA] {problem_id}/{len(problems)} | {problem_name}")

        try:
            rows.append(compute_ela_for_problem(problem, problem_id, problem_name))
        except Exception as e:
            rows.append({
                "problem_type": "LLM",
                "problem_name": problem_name,
                "fid": -200,
                "iid": int(problem_id),
                "llm_problem_id": int(problem_id),
                "FAILED": 1,
                "ERROR": str(e),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved LLM ELA features to: {OUT_FILE}")


if __name__ == "__main__":
    main()
