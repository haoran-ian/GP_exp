import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.min_pop_size = 10
        self.max_pop_size = 40
        self.F = 0.5
        self.CR = 0.9

    def chaotic_initialization(self, lb, ub, size):
        x = np.zeros(size)
        x[0] = np.random.rand()
        for i in range(1, size[0]):
            x[i] = 4 * x[i - 1] * (1 - x[i - 1])
        scaled_x = lb + (ub - lb) * x
        return scaled_x

    def self_adaptive_parameters(self):
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = np.random.uniform(0.1, 0.9)

    def adapt_mutation_scaling(self, fitness):
        avg_fitness = np.mean(fitness)
        self.F = np.clip(0.4 + 0.5 * (avg_fitness / (avg_fitness + 1)), 0.4, 0.9)

    def dynamic_population_size(self, iteration, max_iterations):
        return self.min_pop_size + (self.max_pop_size - self.min_pop_size) * (1 - iteration / max_iterations)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, (self.initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.initial_pop_size
        iteration = 0
        max_iterations = self.budget // self.initial_pop_size

        while num_evaluations < self.budget:
            pop_size = int(self.dynamic_population_size(iteration, max_iterations))
            new_population = np.copy(population[:pop_size])
            for i in range(pop_size):
                self.self_adaptive_parameters()
                self.adapt_mutation_scaling(fitness[:pop_size])
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                if num_evaluations >= self.budget:
                    break

            population = new_population
            iteration += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]