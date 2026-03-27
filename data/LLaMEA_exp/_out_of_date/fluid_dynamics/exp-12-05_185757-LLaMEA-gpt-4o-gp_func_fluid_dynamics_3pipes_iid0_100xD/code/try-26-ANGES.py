import numpy as np

class ANGES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.neighborhood_size = 5
        self.mutation_strength = 0.1
        self.elite_fraction = 0.2

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        budget_used = self.population_size

        while budget_used < self.budget:
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elites = population[elite_indices]

            # Adjust mutation strength based on fitness variance
            fitness_variance = np.var(fitness[elite_indices])
            self.mutation_strength *= (0.9 if fitness_variance < np.mean(fitness) else 1.1) * (1 - (budget_used / self.budget) ** 2)  # Modified line

            self.population_size = int(50 * (1 - budget_used / self.budget))  # Changed line
            self.neighborhood_size = max(1, int(5 * (1 - budget_used / self.budget)))
            neighborhoods = [elites[i] + np.random.normal(0, self.mutation_strength, (self.neighborhood_size, self.dim))
                             for i in range(elite_size)]
            neighborhoods = np.clip(neighborhoods, self.lower_bound, self.upper_bound)

            neighborhood_fitness = [np.apply_along_axis(func, 1, neighborhood) for neighborhood in neighborhoods]
            neighborhood_best_indices = [np.argmin(nf) for nf in neighborhood_fitness]
            neighborhood_best = [neighborhoods[i][neighborhood_best_indices[i]] for i in range(elite_size)]

            population = np.concatenate((elites, neighborhood_best), axis=0)
            fitness = np.array([func(ind) for ind in population])
            budget_used += elite_size * self.neighborhood_size

        best_index = np.argmin(fitness)
        return population[best_index]

# Example usage:
# optimizer = ANGES(budget=1000, dim=10)
# best_solution = optimizer(lambda x: np.sum(x**2))