import numpy as np

class AdaptiveMultimodalEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Crossover probability
        self.diversity_threshold = 0.1

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
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub)
                offspring.append(self.mutate(child, lb, ub))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            self.population_size = self.adaptive_population_resizing(fitness)

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub):
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub):
        mutation_strength = self.random_state.rand() * 0.1
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)

    def adaptive_population_resizing(self, fitness):
        diversity = np.std(fitness)
        if diversity < self.diversity_threshold:
            return min(self.population_size * 2, self.budget // 10)
        else:
            return max(self.initial_population_size, self.population_size // 2)