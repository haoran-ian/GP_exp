## Access Problems
### environment
pyhton=3.8 (mandotory for calculating ELA)
python>=3.11 (mandotory for lens_opt)
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
Here is the simple usage of these problems, for more details about the usage of **ioh**, please check [IOHExperimenter](https://github.com/IOHprofiler/IOHexperimenter/tree/master/example/tutorial.ipynb):
```python
import numpy as np

dim = problem.meta_data.n_variables
name = problem.meta_data.name
lower_bound = problem.bounds.lb
upper_bound = problem.bounds.ub

# you can pass single solution to the problem
x = np.random.uniform(lower_bound, upper_bound)
y = problem(x)
# or a set of solutions
x = np.random.uniform(lower_bound, upper_bound, size=(1000, lower_bound.shape[0]))
y = problem(x)
```
