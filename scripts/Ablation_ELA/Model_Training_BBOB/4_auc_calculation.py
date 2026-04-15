# fmt: off
import os
import sys
import glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
from utils.problems_factory import get_example_problem, ProblemName
# fmt: on

ROOT_DIR = "Benchmark_Top_Results"
SUB_FOLDERS = ["BASELINE", "MEALPY", "OPYTIMIZER"]
OUTPUT_FILE = "data/Ablation_ELA/algorithm_auc_performance.csv"


def calculate_single_run_auc(history, budget):
    if not history:
        return 0.0
    valid_history = [p for p in history if p[0] <= budget]
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
    with open(file_path, 'r') as f:
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
                    current_run.append((int(parts[0]), float(parts[1])))
        if current_run:
            all_runs.append(current_run)
    return all_runs


def process_all_results():
    problem_name_value = {
        "PHOTONIC_2LAYERS_ELLIPSOMETRY": 2,
        "PHOTONIC_10LAYERS_BRAGG": 3,
        "PHOTONIC_20LAYERS_BRAGG": 4,
        "PHOTONIC_10LAYERS_PHOTOVOLTAIC": 5,
    }
    results = []
    for sub in SUB_FOLDERS:
        sub_path = os.path.join(ROOT_DIR, sub)
        if not os.path.exists(sub_path):
            continue
        for folder_name in os.listdir(sub_path):
            folder_path = os.path.join(sub_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            fid, iid, dim, problem_name = None, None, None, None
            algname = ""
            if "_ProblemName." in folder_name:
                parts = folder_name.split("_ProblemName.")
                algname = parts[0]
                problem_name = parts[1]
                try:
                    p_enum = ProblemName(problem_name_value[problem_name])
                    prob_inst = get_example_problem(p_enum)
                    dim = prob_inst.meta_data.n_variables
                except:
                    print(
                        f"Warning: Could not determine dim for {problem_name}")
                    continue
            else:
                try:
                    parts = folder_name.split("_")
                    dim = int(parts[-1].replace("D", ""))
                    iid = int(parts[-2])
                    fid = int(parts[-3][1:])
                    algname = "_".join(parts[:-3])
                except:
                    print(f"Skipping malformed folder: {folder_name}")
                    continue
            budget = dim * 100
            dat_files = glob.glob(os.path.join(folder_path, "*", "*.dat"))
            if not dat_files:
                continue
            dat_path = dat_files[0]
            runs_data = parse_dat_file(dat_path)
            run_aucs = [calculate_single_run_auc(
                run, budget) for run in runs_data]
            if run_aucs:
                results.append({
                    "algname": algname,
                    "fid": fid,
                    "iid": iid,
                    "dim": dim,
                    "problem_name": problem_name if problem_name else f"F{fid}",
                    "auc_mean": np.mean(run_aucs),
                    "auc_std": np.std(run_aucs),
                    "source": sub
                })
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Performance data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_results()
