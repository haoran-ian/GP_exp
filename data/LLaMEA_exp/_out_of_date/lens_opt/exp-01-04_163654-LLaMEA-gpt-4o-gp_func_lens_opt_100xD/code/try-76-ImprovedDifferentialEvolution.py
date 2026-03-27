import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.5
        self.CR = 0.9
        self.levy_scale = 0.1

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

    def levy_flight(self):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / beta)
        return self.levy_scale * step

    def dynamic_population_sizing(self, current_evaluations):
        min_pop_size = 5
        max_pop_size = 20
        reduction_factor = (max_pop_size - min_pop_size) / self.budget
        return int(max_pop_size - reduction_factor * current_evaluations)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            self.pop_size = self.dynamic_population_sizing(num_evaluations)
            new_population = np.copy(population[:self.pop_size])  # Adjust population size
            for i in range(self.pop_size):
                self.self_adaptive_parameters()
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c) + self.levy_flight(), lb, ub)
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

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]