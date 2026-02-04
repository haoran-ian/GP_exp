## Access Problems
### environment
pyhton=3.8 (mandotory)
### tutorial
To access problems, you may need to add below first, depending on where you execute your code:
```python
import os
import sys
sys.path.insert(0, os.getcwd())
```
Import problems:
```python
from problems.lens_opt.problem import get_lens_opt_problem
from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
```
The problems are implemented to fit in **ioh** framework:
```python
import ioh

# meta-surface
problem = get_meta_surface_problem()
# mini-Bragg
problem = get_photonic_problem(num_layers=10, problem_type=PROBLEM_TYPE.BRAGG)
# Bragg
problem = get_photonic_problem(num_layers=20, problem_type=PROBLEM_TYPE.BRAGG)
# ellipsometry
problem = get_photonic_problem(problem_type=PROBLEM_TYPE.ELLIPSOMETRY)
# photovoltaic
problem = get_photonic_problem(num_layers=10, problem_type=PROBLEM_TYPE.PHOTOVOLTAIC)
# lens_opt
problem = get_lens_opt_problem()
```
Here is the simple usage of these problems, for more details about the usage of **ioh**, please check:
```python
import numpy as np

dim = problem.meta_data.n_variables
name = problem.meta_data.name
lower_bound = problem.meta_data.lb
upper_bound = problem.meta_data.ub

x = np.random.uniform(lb, ub)
y = problem(x)
```