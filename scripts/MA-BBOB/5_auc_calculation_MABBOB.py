import os
import glob
import numpy as np
import pandas as pd


ROOT_DIR = "Benchmark_Top_Results_MABBOB"
SUB_FOLDERS = ["BASELINE", "MEALPY", "OPYTIMIZER"]
OUTPUT_FILE = "data/MABBOB/mabbob_algorithm_auc_performance.csv"

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


def parse_mabbob_folder_name(folder_name):
    """
    Expected:
      algname_MABBOB_<instance_id>_D<dim>

    algname 里本身可能带 underscore，例如 modcma_bipop。
    所以从右侧解析。
    """
    parts = folder_name.split("_")

    try:
        dim = int(parts[-1].replace("D", ""))
        mabbob_instance_id = int(parts[-2])
        assert parts[-3] == "MABBOB"
        algname = "_".join(parts[:-3])
    except Exception:
        return None

    return algname, mabbob_instance_id, dim


def process_all_results():
    results = []

    for sub in SUB_FOLDERS:
        sub_path = os.path.join(ROOT_DIR, sub)

        if not os.path.exists(sub_path):
            continue

        for folder_name in os.listdir(sub_path):
            folder_path = os.path.join(sub_path, folder_name)

            if not os.path.isdir(folder_path):
                continue

            parsed = parse_mabbob_folder_name(folder_name)
            if parsed is None:
                print(f"Skipping malformed folder: {folder_name}")
                continue

            algname, mabbob_instance_id, dim = parsed
            budget = dim * 100

            dat_files = glob.glob(os.path.join(folder_path, "*", "*.dat"))
            if not dat_files:
                continue

            dat_path = dat_files[0]
            runs_data = parse_dat_file(dat_path)

            run_aucs = [
                calculate_single_run_auc(run, budget)
                for run in runs_data
            ]
            run_aucs = [x for x in run_aucs if np.isfinite(x)]

            if run_aucs:
                results.append({
                    "algname": algname,
                    "fid": -100,
                    "iid": int(mabbob_instance_id),
                    "dim": int(dim),
                    "mabbob_instance_id": int(mabbob_instance_id),
                    "problem_name": f"MABBOB_{int(mabbob_instance_id)}",
                    "auc_mean": float(np.mean(run_aucs)),
                    "auc_std": float(np.std(run_aucs)),
                    "source": sub,
                })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[Saved] {OUTPUT_FILE}")
    print(df.head())
    print(f"Rows: {len(df)}")
    print(f"Instances: {df[['iid', 'dim']].drop_duplicates().shape[0] if len(df) else 0}")
    print(f"Algorithms: {df['algname'].nunique() if len(df) else 0}")


if __name__ == "__main__":
    process_all_results()