import numpy as np

class EnhancedDynamicPopulationStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr_base = 0.7  # Base crossover probability
        self.mutation_factor = 0.1  # Base mutation factor

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
                parent1, parent2 = self.select_parents(elite, elite_count)
                child = self.crossover(parent1, parent2, lb, ub, np.std(fitness))
                offspring.append(self.mutate(child, lb, ub, np.std(fitness), convergence_rate=evaluations / self.budget))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            # Dynamically adjust population size
            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def select_parents(self, elite, elite_count):
        parent1 = elite[self.random_state.randint(elite_count)]
        parent2 = elite[self.random_state.randint(elite_count)]
        return parent1, parent2

    def crossover(self, parent1, parent2, lb, ub, diversity):
        cr = self.cr_base + 0.3 * diversity  # Adjust crossover probability based on diversity
        mask = self.random_state.rand(self.dim) < cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, diversity, convergence_rate):
        mutation_strength = self.mutation_factor * diversity * (1 - convergence_rate)  # Adjust mutation strength based on diversity and convergence rate
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)