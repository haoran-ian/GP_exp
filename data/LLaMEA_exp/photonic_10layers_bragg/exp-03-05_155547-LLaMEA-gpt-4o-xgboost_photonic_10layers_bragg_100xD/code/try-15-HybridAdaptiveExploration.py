import numpy as np

class HybridAdaptiveExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Crossover probability
        self.exploration_weight = 0.5

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
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub)
                offspring.append(self.mutate(child, lb, ub, evaluations/self.budget))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Hybrid multi-objective strategy: adjust exploration weight
            progress = evaluations / self.budget
            exploration_scale = self.exploration_weight * (1 - progress)
            self.adaptive_exploration(population, lb, ub, exploration_scale)

            # Dynamically adjust population size
            population_size = max(10, int(self.initial_population_size * (1 - progress)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub):
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, progress):
        mutation_strength = self.random_state.rand() * 0.1 * (1 - progress**2)
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)

    def adaptive_exploration(self, population, lb, ub, scale):
        exploration_vector = self.random_state.uniform(lb, ub, size=(population.shape[0], self.dim))
        population += scale * (exploration_vector - population)
        np.clip(population, lb, ub, out=population)