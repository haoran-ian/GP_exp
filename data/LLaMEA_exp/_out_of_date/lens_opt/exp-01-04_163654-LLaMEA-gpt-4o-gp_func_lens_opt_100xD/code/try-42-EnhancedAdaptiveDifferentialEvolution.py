import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.initial_F = 0.5
        self.initial_CR = 0.9
        self.F = self.initial_F
        self.CR = self.initial_CR

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            new_population = np.copy(population)
            if num_evaluations % (self.budget // 10) == 0:
                self.F = self.initial_F * (0.5 + 0.5 * (np.random.rand()))
                self.CR = self.initial_CR * (0.5 + 0.5 * (np.random.rand()))

            for i in range(self.pop_size):
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
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_idx = i
                if num_evaluations >= self.budget:
                    break

            population = new_population

        return population[best_idx], fitness[best_idx]