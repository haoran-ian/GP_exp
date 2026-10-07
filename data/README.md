# Data index

This directory contains raw inputs, processed feature tables, model artifacts, and intermediate experiment outputs.

## Main problem sources

| Location | Contents |
|---|---|
| `ELA/` | Original BBOB ELA calculations, including the 55-feature source tables |
| `Ablation_ELA/` | Processed BBOB ELA pipeline and correlation-filtered tables |
| `MABBOB/` | MA-BBOB instances, ELA features, algorithm performance, and AUC data |
| `LLM/` | LLM-generated problems, ELA features, algorithm performance, and AUC data |
| `Combined/` | Merged training tables, fitted models, feature importance, and feature-selection outputs |

## Benchmark and problem data

| Location | Contents |
|---|---|
| `benchmark_algs/` | Algorithm benchmark outputs |
| `benchmark_funcs/` | Function benchmark outputs |
| `GP_results/` | Genetic-programming experiment results |
| `LLaMEA_exp/` | LLaMEA experiment data |
| `solutions/` | Stored candidate or reference solutions |
| `dimensions/` | Dimension-related inputs |
| `ioh_dat/`, `1000D_y_ioh_dat/` | IOH-formatted experiment data |
| `y/`, `1000D_y/` | Objective-value samples |

## ELA preprocessing distinction

- BBOB feature tables in `Ablation_ELA/Processed_ELA_Pipeline/` have already undergone the existing per-problem correlation-filtering workflow and alignment.
- The inspected MA-BBOB and LLM 55-feature source tables did not undergo an equivalent correlation filter. Constant, all-null, or failed columns may still be removed by downstream training scripts.
- Cross-source experiments should record the exact feature list used instead of assuming that the three sources have identical columns.

Large generated files and fitted model artifacts are intentionally kept under `data/` because existing scripts refer to their current paths.
