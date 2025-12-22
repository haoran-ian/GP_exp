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
        elite_solutions = []
        
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
                    adaptive_step_size *= 1.3  # Slightly increased step size boost
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.85  # Slightly increased reduction on no improvement
                    
                # Archive elite solutions
                if candidate_value < best_value * 1.1:  # Archiving near-best solutions
                    elite_solutions.append(candidate)

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Diversify from elite solutions
            if elite_solutions:
                elite_choice = elite_solutions[np.random.randint(len(elite_solutions))]
                adaptive_step_size = np.maximum((ub - lb) / 20, adaptive_step_size)  # Reset step size with a floor
            
        return best_solution