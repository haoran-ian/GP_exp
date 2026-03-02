import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.exploration_factor = 0.3
        self.exploitation_factor = 0.7
        self.scaling_factor = 0.85  # Slightly improved scaling factor
        self.crossover_rate = 0.7
        self.elite_fraction = 0.2
        self.chaos_control_factor = 0.5

    def chaotic_sequence(self, size):
        # Generate chaotic sequence using logistic map
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 4 * x * (1 - x)  # Logistic map
            chaotic_seq[i] = x
        return chaotic_seq

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

            for i in range(population_size):
                if chaotic_seq[i] < self.exploration_factor:
                    # Exploration: Differential evolution-inspired mutation
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c), lb, ub)
                else:
                    # Exploitation: Crossover with best solution
                    mutant = best_solution + np.random.normal(0, 0.1, self.dim)
                    mutant = np.clip(mutant, lb, ub)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            # Elitist Selection
            sorted_indices = np.argsort(fitness)
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

            # Dynamic population sizing
            if evaluations < self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                population_size = min(population_size * 2, int(self.budget / 10))
                new_members = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                population = np.vstack((population, new_members))
                fitness = np.hstack((fitness, np.apply_along_axis(func, 1, new_members)))
                evaluations += population_size - len(fitness)

        return best_solution