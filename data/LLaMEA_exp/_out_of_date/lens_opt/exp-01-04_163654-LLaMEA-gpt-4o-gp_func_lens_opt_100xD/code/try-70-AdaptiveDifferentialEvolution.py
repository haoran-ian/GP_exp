import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.initial_F = 0.5
        self.initial_CR = 0.9
        self.F_range = (0.1, 0.9)
        self.CR_range = (0.1, 0.9)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        F_values = np.random.uniform(*self.F_range, self.pop_size)
        CR_values = np.random.uniform(*self.CR_range, self.pop_size)
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            new_F_values = np.copy(F_values)
            new_CR_values = np.copy(CR_values)

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F, CR = F_values[i], CR_values[i]
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i], fitness[i] = trial, trial_fitness
                    new_F_values[i] = np.clip(F + np.random.uniform(-0.1, 0.1), *self.F_range)
                    new_CR_values[i] = np.clip(CR + np.random.uniform(-0.1, 0.1), *self.CR_range)

                if num_evaluations >= self.budget:
                    break

            # Reduce population based on fitness
            sorted_indices = np.argsort(fitness)
            population = new_population[sorted_indices]
            fitness = fitness[sorted_indices]
            F_values = new_F_values[sorted_indices]
            CR_values = new_CR_values[sorted_indices]

            if self.pop_size > 5:
                self.pop_size = max(5, self.pop_size - 1)
                population = population[:self.pop_size]
                fitness = fitness[:self.pop_size]
                F_values = F_values[:self.pop_size]
                CR_values = CR_values[:self.pop_size]

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]