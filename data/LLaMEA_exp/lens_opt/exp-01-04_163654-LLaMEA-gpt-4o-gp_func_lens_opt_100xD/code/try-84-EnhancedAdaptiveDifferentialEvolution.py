import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
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

    def levy_flight(self, lb, ub, current_position):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=(self.dim,))
        v = np.random.normal(0, 1, size=(self.dim,))
        step = u / np.abs(v)**(1 / beta)
        return np.clip(current_position + 0.01 * step * (current_position - lb), lb, ub)

    def self_adaptive_parameters(self, success_rate):
        if success_rate > 0.2:
            self.F = np.random.uniform(0.4, 0.9)
            self.CR = np.random.uniform(0.1, 0.9)
        else:
            self.F *= 0.9
            self.CR *= 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.chaotic_initialization(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        success_count = 0

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                success_rate = success_count / (i+1) if i > 0 else 0
                self.self_adaptive_parameters(success_rate)
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
                    success_count += 1
                else:
                    # Levy flight usage for stagnation
                    new_population[i] = self.levy_flight(lb, ub, population[i])
                if num_evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]