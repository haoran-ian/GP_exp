import numpy as np

class MultiPhaseAdaptiveStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)

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
            diversity = np.std(fitness)
            exploration_phase = evaluations < 0.5 * self.budget
            cr = 0.9 if exploration_phase else 0.5  # More exploration early on

            for _ in range(population_size - elite_count):
                parent1, parent2 = self.select_parents(elite, diversity)
                child = self.crossover(parent1, parent2, lb, ub, cr)
                offspring.append(self.mutate(child, lb, ub, diversity, exploration_phase))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def select_parents(self, elite, diversity):
        return elite[self.random_state.randint(len(elite))], elite[self.random_state.randint(len(elite))]

    def crossover(self, parent1, parent2, lb, ub, cr):
        mask = self.random_state.rand(self.dim) < cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, diversity, exploration_phase):
        mutation_strength = (0.1 if exploration_phase else 0.05) * diversity
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)