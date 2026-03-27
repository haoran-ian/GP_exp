import numpy as np

class EnhancedAdaptivePopulationStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.phase_threshold = 0.5  # Split budget into exploration and exploitation phase

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
                child = self.crossover(parent1, parent2, lb, ub)
                offspring.append(self.mutate(child, lb, ub, fitness, evaluations))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            if evaluations / self.budget >= self.phase_threshold:
                population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def select_parents(self, elite, elite_fitness):
        total_fitness = np.sum(elite_fitness)
        probabilities = elite_fitness / total_fitness if total_fitness > 0 else np.ones(len(elite)) / len(elite)
        parent1 = elite[self.random_state.choice(len(elite), p=probabilities)]
        parent2 = elite[self.random_state.choice(len(elite), p=probabilities)]
        return parent1, parent2

    def crossover(self, parent1, parent2, lb, ub):
        cr = 0.8
        mask = self.random_state.rand(self.dim) < cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, fitness, evaluations):
        variance = np.var(fitness)
        mutation_strength = 0.1 * variance / (1 + evaluations / self.budget)
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)