import numpy as np

class ImprovedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.lévy_exponent = 1.5
        self.scaling_factor = 0.8
        self.crossover_rate = 0.7
        self.memory_size = 5

    def chaotic_sequence(self, size):
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 4 * x * (1 - x)
            chaotic_seq[i] = x
        return chaotic_seq

    def lévy_flight(self, size):
        u = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.lévy_exponent)
        return step

    def adaptive_memory(self, fitness_history):
        if len(fitness_history) < self.memory_size:
            return np.mean(fitness_history)
        else:
            return np.mean(fitness_history[-self.memory_size:])

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
                if chaotic_seq[i] < 0.5:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c), lb, ub)
                else:
                    mutant = best_solution + lévy_steps[i] * np.random.normal(0, 0.1, self.dim)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            if np.min(new_fitness) < best_fitness:
                best_fitness = np.min(new_fitness)
                best_solution = new_population[np.argmin(new_fitness)]
                self.scaling_factor *= 0.9  # decrease to exploit best solutions

            # Update population with better solutions
            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            # Memory-based scaling adjustment
            fitness_history.append(best_fitness)
            if evaluations % (self.initial_population_size * 5) == 0:
                self.scaling_factor = self.adaptive_memory(fitness_history)

            # Dynamic population adjustment for enhanced exploration
            if evaluations < self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                population_size = min(population_size * 1.5, int(self.budget / 10))
                new_members = np.random.uniform(lb, ub, (int(population_size) - len(population), self.dim))
                population = np.vstack((population, new_members))
                fitness = np.hstack((fitness, np.apply_along_axis(func, 1, new_members)))
                evaluations += len(new_members)

            # Adjust scaling factor based on standard deviation of fitness
            self.scaling_factor = 0.5 + np.std(fitness) * 0.3

        return best_solution