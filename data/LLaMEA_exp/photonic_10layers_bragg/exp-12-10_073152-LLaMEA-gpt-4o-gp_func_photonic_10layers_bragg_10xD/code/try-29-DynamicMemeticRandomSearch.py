import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class DynamicMemeticRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5  # Initial larger step size
            reduction_factor = 0.5  # Reduction factor for the step size

            # Surrogate model to capture local landscape
            kernel = C(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                # Update surrogate model with new data
                if self.evaluations < self.budget * 0.8:  # Use only in early iterations
                    X_sample = np.array([current_best])
                    y_sample = np.array([current_best_value])
                    if X_sample.shape[0] > 1:
                        gp.fit(X_sample, y_sample)

                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.3  # Increased step size adjustment on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.8  # Adjusted step size reduction on no improvement

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor  # Reduce step size scale
                    no_improvement_count = 0  # Reset count

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Multi-start strategy: restart with reduced step size for refined exploration
            adaptive_step_size = (ub - lb) / 12  # Restart with even smaller step size

        return best_solution