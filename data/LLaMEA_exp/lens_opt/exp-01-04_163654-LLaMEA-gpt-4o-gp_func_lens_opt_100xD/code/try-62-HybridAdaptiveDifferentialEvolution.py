import numpy as np

class HybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        success_mem, fail_mem = [], []

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F_adapt = np.random.normal(self.F, 0.1)
                F_adapt = np.clip(F_adapt, 0, 1)
                mutant = np.clip(a + F_adapt * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    success_mem.append((F_adapt, self.CR))
                else:
                    fail_mem.append((F_adapt, self.CR))
                if num_evaluations >= self.budget:
                    break

            if success_mem:
                self.F = np.mean([f for f, cr in success_mem])
                self.CR = np.mean([cr for f, cr in success_mem])
            elif fail_mem:
                self.F = np.mean([f for f, cr in fail_mem])
                self.CR = np.mean([cr for f, cr in fail_mem])

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]