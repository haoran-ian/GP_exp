import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.base_F = 0.5
        self.base_CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        best_cost = fitness[best_idx]

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = self.base_F + np.random.normal(0, 0.1)  # Dynamic scaling factor
                mutant = np.clip(a + F * (b - c), lb, ub)
                CR = self.base_CR - (fitness[i] - best_cost) / abs(best_cost) if best_cost != 0 else self.base_CR
                CR = np.clip(CR, 0, 1)  # Adaptive crossover rate
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_cost:
                        best_idx = i
                        best_cost = trial_fitness
                if num_evaluations >= self.budget:
                    break

            population = new_population

        return population[best_idx], fitness[best_idx]