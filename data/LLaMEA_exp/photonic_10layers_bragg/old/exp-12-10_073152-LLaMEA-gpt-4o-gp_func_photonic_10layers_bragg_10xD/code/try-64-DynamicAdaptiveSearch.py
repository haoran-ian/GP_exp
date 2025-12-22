import numpy as np

class DynamicAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        population_size = 10  # Dynamic population size
        exploration_ratio = 0.8  # Percentage of budget spent on exploration
        exploration_budget = int(self.budget * exploration_ratio)
        exploitation_budget = self.budget - exploration_budget

        # Phase 1: Exploration with Population-based Search
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        population_values = np.array([func(individual) for individual in population])
        self.evaluations += population_size
        
        for _ in range(exploration_budget // population_size):
            adaptive_step_size = (ub - lb) / 5
            for i in range(population_size):
                candidate = np.clip(population[i] + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < population_values[i]:
                    population[i] = candidate
                    population_values[i] = candidate_value
                    adaptive_step_size *= 1.2
                else:
                    adaptive_step_size *= 0.9

            # Reinitialize a portion of the population with a new random sample based on convergence
            if np.std(population_values) < 0.1:
                num_reinitialize = population_size // 4
                reinitialize_indices = np.random.choice(population_size, num_reinitialize, replace=False)
                population[reinitialize_indices] = np.random.uniform(lb, ub, (num_reinitialize, self.dim))
                population_values[reinitialize_indices] = [func(ind) for ind in population[reinitialize_indices]]
                self.evaluations += num_reinitialize

        # Phase 2: Intensified Local Search
        best_index = np.argmin(population_values)
        best_solution = population[best_index]
        best_value = population_values[best_index]
        
        while self.evaluations < self.budget:
            adaptive_step_size = (ub - lb) / 10
            candidate = np.clip(best_solution + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
            candidate_value = func(candidate)
            self.evaluations += 1

            if candidate_value < best_value:
                best_solution = candidate
                best_value = candidate_value
                adaptive_step_size *= 1.1
            else:
                adaptive_step_size *= 0.95

        return best_solution