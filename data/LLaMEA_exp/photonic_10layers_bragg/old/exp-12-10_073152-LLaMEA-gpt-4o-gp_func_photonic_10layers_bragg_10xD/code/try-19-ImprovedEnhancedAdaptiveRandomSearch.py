import numpy as np

class ImprovedEnhancedAdaptiveRandomSearch:
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
            diversity_threshold = 0.2 * np.mean(ub - lb)  # Diversity threshold for restart
            
            while self.evaluations < self.budget:
                mutation_scale = np.random.uniform(0.5, 1.5)  # Dynamic mutation scaling
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim) * mutation_scale, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.2  # Increased step size adjustment on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.9  # Adjusted step size reduction on no improvement

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor  # Reduce step size scale
                    no_improvement_count = 0  # Reset count
                
                # Check diversity to decide on restart
                if np.linalg.norm(current_best - candidate) < diversity_threshold:
                    break

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Multi-start strategy with diversity-aware condition
            adaptive_step_size = (ub - lb) / 15  # Restart with smaller step size for refined exploration

        return best_solution