import numpy as np

class EnhancedAdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        archive = []
        
        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5
            reduction_factor = 0.5

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.3
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.8

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

                if self.evaluations % (self.budget // 5) == 0:  # Dynamic budget allocation
                    archive.append((current_best, current_best_value))

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            adaptive_step_size = (ub - lb) / 12

            if archive:
                archive_best = min(archive, key=lambda x: x[1])[0]
                if np.random.rand() < 0.5:
                    current_best = archive_best
                    adaptive_step_size = (ub - lb) / 15  # Refined step size from archive solution

        return best_solution