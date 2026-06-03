# fmt: off
import os
import glob
import numpy as np
import pandas as pd
# fmt: on


ROOT_DIR = "Benchmark_Top_Results_LLM"
SUB_FOLDERS = ["BASELINE", "MEALPY", "OPYTIMIZER"]
OUTPUT_FILE = "data/LLM/llm_algorithm_auc_performance.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def calculate_single_run_auc(history, budget):
    if not history:
        return np.nan

    valid_history = [p for p in history if p[0] <= budget]
    if len(valid_history) == 0:
        return np.nan

    area = 0.0
    last_e, last_y = valid_history[0]

    for i in range(1, len(valid_history)):
        curr_e, curr_y = valid_history[i]
        area += (curr_e - last_e) * last_y
        last_e, last_y = curr_e, curr_y

    if last_e < budget:
        area += (budget - last_e + 1) * last_y

    initial_y = valid_history[0][1]
    if initial_y != 0:
        return area / (budget * initial_y)

    return area / budget


def parse_dat_file(file_path):
    all_runs = []
    current_run = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if "evaluations raw_y" in line:
                if current_run:
                    all_runs.append(current_run)
                current_run = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        current_run.append((int(parts[0]), float(parts[1])))
                    except Exception:
                        pass

        if current_run:
            all_runs.append(current_run)

    return all_runs


def parse_llm_folder_name(folder_name):
    """
    Expected:
      algname_LLM_<llm_problem_id>_D<dim>

    algname may contain underscores, e.g. modcma_bipop.
    """
    parts = folder_name.split("_")

    try:
        dim = int(parts[-1].replace("D", ""))
        llm_problem_id = int(parts[-2])
        assert parts[-3] == "LLM"
        algname = "_".join(parts[:-3])
    except Exception:
        return None

    return algname, llm_problem_id, dim


def process_all_results():
    results = []

    for sub in SUB_FOLDERS:
        sub_path = os.path.join(ROOT_DIR, sub)

        if not os.path.exists(sub_path):
            print(f"[Skip] Missing folder: {sub_path}")
            continue

        for folder_name in os.listdir(sub_path):
            folder_path = os.path.join(sub_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            parsed = parse_llm_folder_name(folder_name)
            if parsed is None:
                print(f"[Skip] Cannot parse folder name: {folder_name}")
                continue

            algname, llm_problem_id, dim = parsed
            budget = int(100 * dim)

            dat_files = glob.glob(os.path.join(folder_path, "*.dat"))
            if not dat_files:
                print(f"[Skip] No dat file in: {folder_path}")
                continue

            aucs = []
            for dat_file in dat_files:
                all_runs = parse_dat_file(dat_file)
                for hist in all_runs:
                    auc = calculate_single_run_auc(hist, budget)
                    if np.isfinite(auc):
                        aucs.append(auc)

            if len(aucs) == 0:
                print(f"[Skip] No valid AUC: {folder_path}")
                continue

            results.append({
                "problem_type": "LLM",
                "fid": -200,
                "iid": int(llm_problem_id),
                "llm_problem_id": int(llm_problem_id),
                "dim": int(dim),
                "problem_name": f"LLM_{int(llm_problem_id)}",
                "algname": algname,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "n_runs": int(len(aucs)),
                "source_folder": sub,
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved LLM algorithm AUC performance to: {OUTPUT_FILE}")
    print(df.head())
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    process_all_results()
