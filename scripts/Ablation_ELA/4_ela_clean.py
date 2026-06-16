# fmt: off
import os
import re
import sys
import glob
import warnings
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
from utils.problems_factory import get_example_problem, ProblemName
# fmt: on


ATOM_DIR = "data/Ablation_ELA/atom"
OUTPUT_DIR = "data/Ablation_ELA/Processed_ELA_Pipeline"
FINAL_CSV = os.path.join(OUTPUT_DIR, "pipeline_aligned_ela.csv")


def remove_redundant_features(df, threshold=0.95):
    meta_cols = ['problem_name', 'fid', 'iid', 'dim', 'seed']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols]
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(
        upper[column] > threshold)]
    print(
        f"  [-] Dropped {len(to_drop)} highly correlated features (threshold > {threshold}).")
    cleaned_df = df.drop(columns=to_drop)
    return cleaned_df, (to_drop)


def parse_filename(filename):
    problem_name_value = {
        "PHOTONIC_2LAYERS_ELLIPSOMETRY": 2,
        "PHOTONIC_10LAYERS_BRAGG": 3,
        "PHOTONIC_20LAYERS_BRAGG": 4,
        "PHOTONIC_10LAYERS_PHOTOVOLTAIC": 5,
    }
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("-")
    metadata = {}
    try:
        if parts[0].startswith('F') and parts[0][1:].isdigit():
            metadata['fid'] = int(parts[0][1:])
            metadata['iid'] = int(parts[1])
            metadata['dim'] = int(parts[2][1:])
            metadata['ela_set'] = parts[3]
            metadata['problem_name'] = f"BBOB_F{metadata['fid']}"
        else:
            problem_name = parts[0].split(".")[1]
            metadata['problem_name'] = problem_name
            metadata['ela_set'] = int(parts[1])
            metadata['fid'] = -1
            metadata['iid'] = -1
            p_enum = ProblemName(problem_name_value[problem_name])
            prob_inst = get_example_problem(p_enum)
            dim = prob_inst.meta_data.n_variables
            metadata['dim'] = dim
        for p in parts:
            if "seed:" in p:
                metadata['seed'] = int(p.split(":")[1])
                break
    except Exception as e:
        return None

    return metadata


def run_ela_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(ATOM_DIR, "*.csv"))
    aggregated_data = {}
    print(f"Scanning {len(all_files)} files...")
    counter = 0
    for f in all_files:
        if not f.startswith("data/Ablation_ELA/atom/ProblemName.PHOTONIC_20LAYERS_BRAGG"):
            continue
        counter += 1
        if counter % 1000 == 0:
            print(counter)
        meta = parse_filename(f)
        if not meta:
            continue
        instance_key = (meta['problem_name'], meta['iid'],
                        meta['dim'], meta['seed'])
        if instance_key not in aggregated_data:
            aggregated_data[instance_key] = {
                'problem_name': meta['problem_name'],
                'fid': meta['fid'],
                'iid': meta['iid'],
                'dim': meta['dim'],
                'seed': meta['seed']
            }
        try:
            df = pd.read_csv(f)
            feat_dict = df.iloc[0].to_dict()
            feat_dict = {k: v for k, v in feat_dict.items()
                         if not k.startswith('Unnamed')}
            aggregated_data[instance_key].update(feat_dict)
        except:
            continue
    full_df = pd.DataFrame.from_dict(aggregated_data, orient='index')
    cleaned_df, dropped_list = remove_redundant_features(full_df)
    cleaned_df.to_csv(FINAL_CSV, index=False)
    meta_cols = ['problem_name', 'fid', 'iid', 'dim', 'seed']
    feature_cols = [c for c in cleaned_df.columns if c not in meta_cols]
    print(f"Total raw features found: {len(feature_cols)}")
    print(f"Pipeline update complete. Saved to {FINAL_CSV}")
    print(f"Total instances: {len(cleaned_df)}")

    return full_df


if __name__ == "__main__":
    run_ela_pipeline()
