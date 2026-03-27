import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
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

    def opposition_based_learning(self, population, lb, ub):
        opposite_population = lb + ub - population
        return opposite_population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            opposite_population = self.opposition_based_learning(population, lb, ub)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            num_evaluations += self.pop_size

            for i in range(self.pop_size):
                self.self_adaptive_parameters()
                idxs = [idx for idx in range(self.pop_size) if idx != i]
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
                if opposite_fitness[i] < fitness[i]:
                    new_population[i] = opposite_population[i]
                    fitness[i] = opposite_fitness[i]
                if num_evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]