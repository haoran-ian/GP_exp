import numpy as np

class RefinedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.elite_fraction = 0.2
        self.stochastic_ranking_probability = 0.45

    def chaotic_sequence(self, size):
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 4 * x * (1 - x)
            chaotic_seq[i] = x
        return chaotic_seq

    def adaptive_mutation(self, a, b, c, lb, ub):
        if np.random.rand() > 0.5:
            return np.clip(a + self.mutation_factor * (b - c), lb, ub)
        else:
            return np.clip(a + self.mutation_factor * (np.random.rand(self.dim) * (ub - lb)), lb, ub)

    def stochastic_ranking(self, fitness, constraints, indices):
        sorted_indices = np.argsort(fitness)
        for i in range(len(fitness) - 1):
            if np.random.rand() > self.stochastic_ranking_probability or constraints[sorted_indices[i]] <= constraints[sorted_indices[i + 1]]:
                if fitness[sorted_indices[i]] > fitness[sorted_indices[i + 1]]:
                    sorted_indices[i], sorted_indices[i + 1] = sorted_indices[i + 1], sorted_indices[i]
        return sorted_indices

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            chaotic_seq = self.chaotic_sequence(population_size)
            constraints = np.zeros(population_size)  # Placeholder for constraint handling

            for i in range(population_size):
                if chaotic_seq[i] < 0.5:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = self.adaptive_mutation(a, b, c, lb, ub)
                else:
                    mutant = best_solution + np.random.normal(0, 0.1, self.dim)
                    mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            sorted_indices = self.stochastic_ranking(new_fitness, constraints, np.arange(population_size))
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = sorted_indices[:elite_count]

            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            elite_population = population[elite_indices]
            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = population[np.argmin(fitness)]

        return best_solution