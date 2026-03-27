import numpy as np

class PhaseTransitionAdaptiveStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr_initial = 0.9  # Initial crossover probability

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
            diversity = np.std(population, axis=0).mean()
            cr = self.cr_initial * (1 - diversity)  # Adjust crossover based on diversity
            mutation_strength = 0.1 * diversity  # Adjust mutation strength

            for _ in range(population_size - elite_count):
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub, cr)
                offspring.append(self.mutate(child, lb, ub, mutation_strength))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            # Dynamically adjust population size with phase transition inspired strategy
            new_population_size = max(10, int(population_size * (1 - evaluations/self.budget) + 5 * np.sin(2 * np.pi * evaluations/self.budget)))
            if new_population_size < population_size:
                indices_to_keep = np.argsort(fitness)[:new_population_size]
                population = population[indices_to_keep]
                fitness = fitness[indices_to_keep]
            population_size = new_population_size

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub, cr):
        mask = self.random_state.rand(self.dim) < cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, mutation_strength):
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)