import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.F = 0.5
        self.CR = 0.9

    def local_search(self, individual, func, lb, ub):
        epsilon = 0.01 * (ub - lb)
        perturbation = np.random.uniform(-epsilon, epsilon, size=self.dim)
        new_point = np.clip(individual + perturbation, lb, ub)
        new_fitness = func(new_point)
        return new_point, new_fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                self.F = 0.5 + 0.5 * np.random.rand()  # Dynamic F
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
                else:
                    # Apply local search to enhance exploitation
                    local_point, local_fitness = self.local_search(trial, func, lb, ub)
                    num_evaluations += 1
                    if local_fitness < fitness[i]:
                        new_population[i] = local_point
                        fitness[i] = local_fitness

                if num_evaluations >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]