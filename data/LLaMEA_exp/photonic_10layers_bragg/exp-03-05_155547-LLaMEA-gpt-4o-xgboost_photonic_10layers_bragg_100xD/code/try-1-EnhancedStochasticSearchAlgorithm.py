import numpy as np

class EnhancedStochasticSearchAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_fraction = 0.2
        self.diversity_threshold = 0.1
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

            offspring = self.generate_offspring(elite, lb, ub, func)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            if self.calculate_diversity(population) < self.diversity_threshold:
                population = self.random_state.uniform(lb, ub, size=(self.population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                evaluations += self.population_size

        best_index = np.argmin(fitness)
        return population[best_index]

    def generate_offspring(self, elite, lb, ub, func):
        offspring = []
        for _ in range(self.population_size - len(elite)):
            parent = elite[self.random_state.randint(len(elite))]
            mutant = self.adaptive_mutate(parent, lb, ub, func)
            offspring.append(mutant)
        return offspring

    def adaptive_mutate(self, parent, lb, ub, func):
        current_best = np.min([func(ind) for ind in parent])
        mutation_strength = (1 - current_best / self.budget) * 0.1
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = parent + noise
        return np.clip(mutant, lb, ub)

    def calculate_diversity(self, population):
        mean_vector = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - mean_vector, axis=1))
        return diversity