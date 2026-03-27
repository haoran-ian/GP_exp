import numpy as np

class AdaptiveStochasticSearchAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.random_state.uniform(lb, ub, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            offspring = []
            for _ in range(self.population_size - elite_count):
                parent = elite[self.random_state.randint(elite_count)]
                offspring.append(self.mutate(parent, lb, ub))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

        best_index = np.argmin(fitness)
        return population[best_index]

    def mutate(self, parent, lb, ub):
        mutation_strength = self.random_state.rand() * 0.1  # Dynamic mutation strength
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = parent + noise
        return np.clip(mutant, lb, ub)