import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.pop_size = self.initial_pop_size
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.3, 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            success_count = 0
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1
                if num_evaluations >= self.budget:
                    break

            if success_count > self.pop_size / 2:
                self.pop_size = min(self.pop_size + 1, self.initial_pop_size * 2)
            elif success_count < self.pop_size / 4:
                self.pop_size = max(self.pop_size - 1, self.initial_pop_size)

            population = new_population
            if num_evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]