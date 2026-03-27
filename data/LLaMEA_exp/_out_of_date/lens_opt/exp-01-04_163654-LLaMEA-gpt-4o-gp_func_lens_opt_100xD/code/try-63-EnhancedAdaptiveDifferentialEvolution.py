import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F_base = 0.5
        self.CR_base = 0.9
        self.delta = 0.1

    def adapt_parameters(self, success_rate):
        self.F = self.F_base + np.random.uniform(-self.delta, self.delta) * success_rate
        self.CR = self.CR_base + np.random.uniform(-self.delta, self.delta) * success_rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        success_count = 0

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                neighbors = np.random.choice(idxs, size=min(5, len(idxs)), replace=False)
                a, b, c = population[neighbors[:3]]
                self.adapt_parameters(success_count / (self.pop_size if self.pop_size else 1))
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
                if num_evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]