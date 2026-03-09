import numpy as np

class HybridAdaptiveStochasticSearchAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_fraction = 0.2
        self.mutation_adapt_rate = 0.05  # Adaptation rate for mutation
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
            
            # Dynamic elitism: vary elite fraction based on fitness variance
            fitness_variance = np.var(fitness)
            self.elite_fraction = max(0.1, min(0.5, fitness_variance))

            offspring = []
            for _ in range(self.population_size - elite_count):
                parent = self.select_parent(elite, fitness[elite_indices])
                offspring.append(self.mutate(parent, lb, ub, fitness_variance))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

        best_index = np.argmin(fitness)
        return population[best_index]

    def select_parent(self, elite, fitness):
        # Tournament selection among elite
        participants = self.random_state.choice(elite, size=3, replace=False)
        participant_fitness = np.array([fitness[np.where(elite == p)[0][0]] for p in participants])
        return participants[np.argmin(participant_fitness)]

    def mutate(self, parent, lb, ub, fitness_variance):
        # Adaptive mutation strength based on fitness variance
        mutation_strength = np.clip(self.random_state.rand() * (0.1 + self.mutation_adapt_rate * fitness_variance), 0.01, 0.2)
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = parent + noise
        return np.clip(mutant, lb, ub)