import numpy as np

class AdaptiveMultiLevelRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        initial_step_size = (ub - lb) / 5
        reduction_factor = 0.6
        min_step_size = (ub - lb) / 100
        
        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 4
            adaptive_step_size = initial_step_size

            while self.evaluations < self.budget and adaptive_step_size > min_step_size:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.2
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.9

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Fine-tuning phase with preserved elite solutions
            elite_candidates = [best_solution]
            elite_step_size = initial_step_size / 10
            for _ in range(3):  # Elite fine-tuning iterations
                for elite in elite_candidates:
                    if self.evaluations >= self.budget:
                        break
                    candidate = np.clip(elite + np.random.uniform(-elite_step_size, elite_step_size, self.dim), lb, ub)
                    candidate_value = func(candidate)
                    self.evaluations += 1
                    if candidate_value < best_value:
                        best_solution = candidate
                        best_value = candidate_value

        return best_solution