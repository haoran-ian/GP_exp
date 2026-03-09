import numpy as np

class EnhancedAdaptivePopulationStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Initial Crossover probability
        self.evaluation_threshold = budget * 0.5  # Threshold to switch strategy

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
                child = self.crossover(parent1, parent2, lb, ub, np.var(fitness))
                offspring.append(self.mutate(child, lb, ub, np.var(fitness), evaluations))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Dynamically adjust population size and strategy
            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))
            if evaluations > self.evaluation_threshold:
                self.elite_fraction = 0.3  # Increase exploitation phase

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub, fitness_variance):
        self.cr = 0.5 + 0.4 * (1 - fitness_variance / np.max((fitness_variance, 1e-6)))  # Adjust crossover probability
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, fitness_variance, evaluations):
        phase_factor = 1 if evaluations < self.evaluation_threshold else 0.5  # Reduce mutation strength in exploitation phase
        mutation_strength = self.random_state.rand() * 0.1 * fitness_variance * phase_factor
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)