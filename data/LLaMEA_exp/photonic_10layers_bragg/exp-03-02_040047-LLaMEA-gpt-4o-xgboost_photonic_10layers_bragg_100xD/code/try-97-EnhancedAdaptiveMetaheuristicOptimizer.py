import numpy as np

class EnhancedAdaptiveMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.lévy_exponent = 1.5
        self.chaos_control_factor = 0.5
        self.scaling_factor = 0.8
        self.crossover_rate = 0.7
        self.memory_size = 5
        self.exploration_factor = 0.3  # Adjusted factor for exploration
        self.mutation_rate = 0.9  # New mutation rate for diversity
        self.dynamic_scale_adaptation = 0.1  # New factor for scaling adaptation

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

    def enhanced_exploration(self, progress):
        return self.exploration_factor / (1 + progress)

    def adaptive_scaling(self, evaluations, budget):
        return self.dynamic_scale_adaptation * (1 - evaluations / budget)

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
            lévy_steps = self.lévy_flight((population_size, self.dim))

            for i in range(population_size):
                if chaotic_seq[i] < self.chaos_control_factor:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c) * self.mutation_rate, lb, ub)
                else:
                    enhanced_factor = self.enhanced_exploration(evaluations / self.budget)
                    mutant = (best_solution + lévy_steps[i] * np.random.normal(0, 0.1, self.dim) 
                              * enhanced_factor)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            if np.min(new_fitness) < best_fitness:
                best_fitness = np.min(new_fitness)
                best_solution = new_population[np.argmin(new_fitness)]

            for i in range(population_size):
                if new_fitness[i] < fitness[i] or np.random.rand() < self.adaptive_scaling(evaluations, self.budget):
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            fitness_history.append(best_fitness)

            if evaluations % (self.initial_population_size * 5) == 0:
                self.scaling_factor = np.std(fitness) * 0.5

            if evaluations < self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                population_size = min(population_size * 2, int(self.budget / 10))
                new_members = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                population = np.vstack((population, new_members))
                fitness = np.hstack((fitness, np.apply_along_axis(func, 1, new_members)))
                evaluations += population_size - len(fitness)

        return best_solution