import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.25
        self.lévy_exponent = 1.7
        self.chaos_control_factor = 0.6
        self.scaling_factor = 0.85
        self.crossover_rate = 0.8
        self.memory_size = 5
        self.tunneling_factor = 0.1 

    def chaotic_sequence(self, size):
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 3.8 * x * (1 - x)
            chaotic_seq[i] = x
        return chaotic_seq

    def lévy_flight(self, size):
        u = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.lévy_exponent)
        return step

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
            lévy_steps = self.lévy_flight(population_size)

            for i in range(population_size):
                if chaotic_seq[i] < self.chaos_control_factor:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c), lb, ub)
                else:
                    tunnel_effect = np.random.normal(0, self.tunneling_factor, self.dim)
                    mutant = best_solution + lévy_steps[i] * np.random.normal(0, 0.1, self.dim) + tunnel_effect

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            sorted_indices = np.argsort(fitness)
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = sorted_indices[:elite_count]

            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            if np.min(new_fitness) < best_fitness:
                best_fitness = np.min(new_fitness)
                best_solution = new_population[np.argmin(new_fitness)]

            fitness_history.append(best_fitness)

            if evaluations % (self.initial_population_size * 5) == 0:
                self.scaling_factor = max(0.1, self.scaling_factor * 0.95)

            if evaluations < self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                population_size = min(population_size * 2, int(self.budget / 10))
                new_members = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                population = np.vstack((population, new_members))
                fitness = np.hstack((fitness, np.apply_along_axis(func, 1, new_members)))
                evaluations += population_size - len(fitness)

        return best_solution