# GP experiment workspace

This workspace studies algorithm performance and ELA features across BBOB, MA-BBOB, LLM-generated functions, and real-world optimization problems.

## Workspace map

| Location | Purpose |
|---|---|
| `problems/` | Real-world optimization problem implementations |
| `scripts/` | Experiment entry points, grouped by workflow and data source |
| `utils/` | Shared helper code |
| `data/` | Raw data, processed features, training tables, and model artifacts |
| `results/` | Figures, tables, ablation results, and shareable reports |
| `experiments/` | Current and archived LLaMEA experiment runs |
| `vendor/` | Local LLaMEA, ModDE, and historical XAI/LLaMEA source trees |
| `artifacts/runtime/` | Runtime artifacts such as Ray state |
| `logs/` | Console and optimizer logs |

Detailed indexes are available in:

- `data/README.md`
- `scripts/README.md`
- `results/README.md`

## Current shareable materials

| Topic | Main file |
|---|---|
| Representative algorithm selection | `results/algorithm_selection_by_principle/selection_report_en.md` |
| Short algorithm-selection email | `results/algorithm_selection_by_principle/email_draft.txt` |
| ELA feature-group design | `results/ela_feature_sets_for_ablation/README.md` |
| BBOB feature groups for email | `results/ela_feature_sets_for_ablation/bbob_feature_groups_for_email.txt` |

## Python environments

The ELA calculation workflow requires Python 3.8. The lens optimization implementation requires Python 3.11 or newer. Use separate environments when running both workflows.

## Using the problem implementations

Run scripts from the workspace root, or add the workspace root to `sys.path`:

```python
import os
import sys

sys.path.insert(0, os.getcwd())
```

Import the real-world problems with:

```python
from problems.lens_opt.problem import get_lens_opt_problem
from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
```

The problems follow the IOH interface:

```python
import numpy as np

problem = get_photonic_problem(num_layers=10, problem_type=PROBLEM_TYPE.BRAGG)

dimension = problem.meta_data.n_variables
x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
y = problem(x)
```

## Organization rules for new work

- Put source-specific intermediate data under the existing source directory, such as `data/Ablation_ELA`, `data/MABBOB`, or `data/LLM`.
- Put cross-source training artifacts under `data/Combined/`.
- Put every new analysis under a named `results/<study_name>/` directory.
- Keep scripts in the matching workflow directory and reusable functions in `utils/`.
- Do not add new experiment dumps at the workspace root.

Timestamped experiment runs are stored in `experiments/archive/`; current generated-algorithm AUC files are stored in `experiments/current/`. Benchmark outputs are under `results/benchmarks/`, and older standalone analyses are under `results/legacy/`.
