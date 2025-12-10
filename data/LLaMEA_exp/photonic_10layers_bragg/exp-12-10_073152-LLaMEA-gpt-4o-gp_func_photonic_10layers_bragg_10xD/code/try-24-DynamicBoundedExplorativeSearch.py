import numpy as np

class DynamicBoundedExplorativeSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = np.random.uniform(lb, ub)
        best_value = func(best_solution)
        self.evaluations += 1
        
        adaptive_step_size = (ub - lb) / 5
        reduction_factor = 0.5
        expansion_factor = 1.3
        no_improvement_count = 0
        max_no_improvement = 5
        exploration_factor = 0.3  # Factor to scale exploration steps

        learned_bounds_lb = lb.copy()
        learned_bounds_ub = ub.copy()

        while self.evaluations < self.budget:
            candidate = np.clip(best_solution + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), learned_bounds_lb, learned_bounds_ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            
            if candidate_value < best_value:
                best_solution = candidate
                best_value = candidate_value
                no_improvement_count = 0
                adaptive_step_size *= expansion_factor
                # Update learned bounds based on new best
                learning_adjustment = exploration_factor * (ub - lb) / (self.evaluations / self.budget + 1)
                learned_bounds_lb = np.maximum(learned_bounds_lb, best_solution - learning_adjustment)
                learned_bounds_ub = np.minimum(learned_bounds_ub, best_solution + learning_adjustment)
            else:
                no_improvement_count += 1
                adaptive_step_size *= 0.8

            if no_improvement_count >= max_no_improvement:
                adaptive_step_size *= reduction_factor
                no_improvement_count = 0

        return best_solution