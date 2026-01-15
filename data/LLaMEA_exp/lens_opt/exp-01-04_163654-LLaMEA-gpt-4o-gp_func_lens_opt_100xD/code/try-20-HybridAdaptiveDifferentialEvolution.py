import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.5
        self.CR = 0.9

    def _mutation_strategy(self, population, i, lb, ub):
        idxs = [idx for idx in range(self.pop_size) if idx != i]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        rand_idx = np.random.randint(self.dim)
        # Allow learning-based strategy selection
        if np.random.rand() < 0.5:
            mutant = a + self.F * (b - c)
        else:
            best = population[np.argmin([func(ind) for ind in population])]
            mutant = a + self.F * (best - a) + self.F * (b - c)
        return np.clip(mutant, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                mutant = self._mutation_strategy(population, i, lb, ub)
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