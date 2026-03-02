import numpy as np

class RefinedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.scaling_factor = 0.8
        self.crossover_rate = 0.7
        self.memory_size = 5
        self.exploration_factor = 0.5

    def chaotic_sequence(self, size):
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 4 * x * (1 - x)
            chaotic_seq[i] = x
        return chaotic_seq

    def adaptive_basin_mapping(self, fitness, population, scaling_factor):
        min_fitness = np.min(fitness)
        max_fitness = np.max(fitness)
        normalized_fitness = (fitness - min_fitness) / (max_fitness - min_fitness + 1e-10)
        basin_map = np.exp(-normalized_fitness / scaling_factor)
        return basin_map / np.sum(basin_map)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        fitness_history = [best_fitness]

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            chaotic_seq = self.chaotic_sequence(population_size)

            basin_map = self.adaptive_basin_mapping(fitness, population, self.scaling_factor)

            for i in range(population_size):
                if chaotic_seq[i] < self.exploration_factor:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c) * basin_map[i], lb, ub)
                else:
                    indices = np.random.choice(population_size, p=basin_map)
                    mutant = population[indices] + np.random.normal(0, 0.1, self.dim)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            if np.min(new_fitness) < best_fitness:
                best_fitness = np.min(new_fitness)
                best_solution = new_population[np.argmin(new_fitness)]
            
            fitness_history.append(best_fitness)
            if evaluations % (self.initial_population_size * 5) == 0:
                self.scaling_factor = np.std(fitness) * 0.5

        return best_solution