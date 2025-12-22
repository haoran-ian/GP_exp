import numpy as np

class EnhancedAdaptiveRandomSearchV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Initial configuration
        population_size = 5
        adaptive_step_size = (ub - lb) / 5
        reduction_factor = 0.25
        max_no_improvement = 5
        exploration_factor = 0.1
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        population_values = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        
        while self.evaluations < self.budget:
            # Sort population by their fitness
            sorted_indices = np.argsort(population_values)
            population = population[sorted_indices]
            population_values = population_values[sorted_indices]
            current_best = population[0]
            current_best_value = population_values[0]
            no_improvement_count = 0
            
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                
                candidate = np.clip(population[i] + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < population_values[i]:
                    population[i] = candidate
                    population_values[i] = candidate_value
                    if candidate_value < current_best_value:
                        current_best = candidate
                        current_best_value = candidate_value
                        no_improvement_count = 0
                        adaptive_step_size = min((ub - lb) * exploration_factor, adaptive_step_size * 1.2)
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.85

                # Dynamic step size reduction and exploration
                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0
                    reduction_factor *= 1.05

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

        return best_solution