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
        
        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5  # Initial larger step size
            reduction_factor = 0.7  # Reduction factor for the step size
            
            # Introducing different scales of perturbation
            large_perturbation_scale = (ub - lb) / 8
            small_perturbation_scale = (ub - lb) / 20
            perturbation_counter = 0

            while self.evaluations < self.budget:
                if perturbation_counter % 2 == 0:
                    perturbation_scale = large_perturbation_scale
                else:
                    perturbation_scale = small_perturbation_scale

                candidate = np.clip(current_best + np.random.uniform(-perturbation_scale, perturbation_scale, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.1  # Slightly increase step size on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.95  # Slightly reduce step size on no improvement

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor  # Reduce step size scale
                    no_improvement_count = 0  # Reset count
                    perturbation_counter += 1

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Multi-start strategy: restart with a dynamically sized step size
            dynamic_step_size = (ub - lb) / (10 + np.random.rand() * 5)  # Restart with a dynamically chosen step size
            adaptive_step_size = dynamic_step_size

        return best_solution