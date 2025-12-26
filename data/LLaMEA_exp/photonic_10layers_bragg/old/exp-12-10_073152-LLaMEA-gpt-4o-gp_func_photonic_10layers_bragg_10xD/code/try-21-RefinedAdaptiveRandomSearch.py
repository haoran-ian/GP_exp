import numpy as np

class RefinedAdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Memory for tracking progress
        global_best = None
        global_best_value = float('inf')
        adaptive_step_memory = []

        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5  # Initial larger step size
            reduction_factor = 0.5  # Reduction factor for the step size

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
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

                # Memory-based step size adjustment
                adaptive_step_memory.append(adaptive_step_size)
                if len(adaptive_step_memory) > 10:
                    adaptive_step_memory.pop(0)

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Global best memory update
            if current_best_value < global_best_value:
                global_best = current_best
                global_best_value = current_best_value

            # Multi-start strategy with strategic restarts
            if np.mean(adaptive_step_memory) < (ub - lb) / 15:
                # If step size becomes too small, trigger a strategic restart
                current_best = np.random.uniform(lb, ub)
                adaptive_step_size = (ub - lb) / 10
                adaptive_step_memory.clear()

        return best_solution