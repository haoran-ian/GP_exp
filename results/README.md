# Results index

This directory contains analysis outputs, figures, tables, and shareable reports.

## Current reports

### Algorithm selection by principle

Directory: `algorithm_selection_by_principle/`

- `selection_report_en.md`: rationale, paper basis, and classification of the selected algorithms.
- `algorithm_inventory_41.csv`: inventory of the 41 candidate algorithms.
- `recommended_8.json`: machine-readable list of the 8 representative algorithms.
- `email_draft.txt`: concise text for sending the algorithm selection to a supervisor.

### ELA feature groups for ablation

Directory: `ela_feature_sets_for_ablation/`

- `README.md`: grouping rationale and recommended ablation protocol.
- `feature_sets.json`: BBOB and MA-BBOB/LLM partitions into 5, 10, 15, and 20 groups.
- `feature_sets_long.csv`: long-form group membership table.
- `bbob_feature_groups_for_email.txt`: concise BBOB grouping text for email.
- `build_feature_groupings.py`: reproducible grouping generator and validator.

## Experiment result families

| Location | Contents |
|---|---|
| `Combined/` | Combined-source prediction, feature-set ablation, and real-world validation results |
| `ela_space_compare/` | BBOB, MA-BBOB, and LLM ELA-space comparisons |
| `ela_convergence/` | ELA convergence results for real-world photonic problems |
| `benchmarks/bbob/` | BBOB algorithm benchmark outputs |
| `benchmarks/mabbob/` | MA-BBOB algorithm benchmark outputs |
| `benchmarks/llm/` | LLM-problem algorithm benchmark outputs |
| `legacy/` | Older MA-BBOB ELA and ablation output directories retained for reproducibility |

## Legacy top-level figures

PNG and CSV files directly under `results/` are outputs from earlier plotting and ablation scripts. They remain in place because several scripts write to or refer to these paths. New analyses should use a named subdirectory instead of adding further files at the root of `results/`.

## Naming convention for new work

Use:

```text
results/<study_name>/
├── README.md
├── tables/
├── figures/
└── artifacts/
```

Small studies may keep files directly inside their named directory, as done for the two current reports above.
