# Scripts index

Scripts are organized by experimental stage or data source.

| Location | Purpose |
|---|---|
| `1_ela_prepare.py` to `9_algorithm_generation_xgboost.py` | Original end-to-end workflow |
| `Ablation_ELA/` | BBOB ELA sampling, cleaning, performance calculation, and ablation |
| `MA-BBOB/` | MA-BBOB generation, selection, performance, AUC, and ablation |
| `LLaMEA_problem/` | LLM-problem ELA, algorithm performance, source comparison, and real-world validation |
| `regressor_models/` | Current regressor comparison, feature selection, and feature pruning scripts |
| `Plots/` | Plotting and solution visualization |

## Current ELA-to-performance workflow

The relevant stages are:

1. Calculate or load ELA features for BBOB, MA-BBOB, and LLM problems.
2. Calculate algorithm performance and normalized AUC targets.
3. Train source-specific or combined regressors.
4. Run feature-family or feature-group ablations.
5. Save analysis outputs under `results/Combined/` or another named results directory.

## Conventions for new scripts

- Put reusable code in `utils/`.
- Put study-specific entry points in the matching source directory.
- Use ordered numeric prefixes only when scripts form a required sequence.
- Write outputs to a named directory under `results/` or `data/<source>/`.
- Add a short README when a workflow needs more than one command.

Some scripts under `LLaMEA_problem/` were previously moved to `regressor_models/`. Their current Git deletion/addition state has intentionally been left untouched.
