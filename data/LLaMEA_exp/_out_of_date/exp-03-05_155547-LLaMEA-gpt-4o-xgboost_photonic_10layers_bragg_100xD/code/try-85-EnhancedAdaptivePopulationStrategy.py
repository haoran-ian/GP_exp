import numpy as np

class EnhancedAdaptivePopulationStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = self.random_state.uniform(lb, ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            offspring = []
            for _ in range(population_size - elite_count):
                parent1, parent2 = self.select_parents(elite, fitness[elite_indices])
                child = self.crossover(parent1, parent2, lb, ub, np.std(fitness))
                offspring.append(self.dynamic_mutate(child, lb, ub, evaluations))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Dynamically adjust population size
            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def select_parents(self, elite, elite_fitness):
        distances = np.sum((elite - elite.mean(axis=0))**2, axis=1)
        prob_selection = distances / np.sum(distances)
        indices = self.random_state.choice(len(elite), size=2, p=prob_selection)
        return elite[indices[0]], elite[indices[1]]

    def crossover(self, parent1, parent2, lb, ub, diversity):
        self.cr = 0.7 + 0.2 * diversity  # Adjust crossover probability based on diversity
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def dynamic_mutate(self, individual, lb, ub, evaluations):
        mutation_strength = 0.1 * (1 - evaluations / self.budget)
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)