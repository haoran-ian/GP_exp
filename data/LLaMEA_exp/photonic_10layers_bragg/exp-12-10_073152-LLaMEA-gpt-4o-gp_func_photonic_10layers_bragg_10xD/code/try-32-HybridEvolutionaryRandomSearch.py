import numpy as np

class HybridEvolutionaryRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        population_size = max(4, self.dim)  # Ensure at least a small population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        pop_values = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        adaptive_step_size = (ub - lb) / 5
        max_no_improvement = 5
        no_improvement_count = 0

        while self.evaluations < self.budget:
            # Select parents and apply crossover
            parents_idx = np.argsort(pop_values)[:2]
            best_parents = population[parents_idx]
            offspring = (best_parents[0] + best_parents[1]) / 2

            # Apply mutation with adaptive step size
            mutation = np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim)
            offspring = np.clip(offspring + mutation, lb, ub)
            offspring_value = func(offspring)
            self.evaluations += 1

            # Selection step: replace worst individual if better
            if offspring_value < np.max(pop_values):
                worst_idx = np.argmax(pop_values)
                population[worst_idx] = offspring
                pop_values[worst_idx] = offspring_value

            # Update best solution found
            if offspring_value < best_value:
                best_solution = offspring
                best_value = offspring_value
                no_improvement_count = 0
                adaptive_step_size *= 1.1  # Increase step size on improvement
            else:
                no_improvement_count += 1
                adaptive_step_size *= 0.9  # Decrease step size on no improvement

            # Restart strategy for exploration
            if no_improvement_count >= max_no_improvement:
                new_individuals = np.random.uniform(lb, ub, (population_size // 2, self.dim))
                population[:population_size // 2] = new_individuals
                pop_values[:population_size // 2] = [func(ind) for ind in new_individuals]
                self.evaluations += population_size // 2
                adaptive_step_size = (ub - lb) / 10
                no_improvement_count = 0

        return best_solution