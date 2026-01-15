import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.pop_size = self.initial_pop_size
        self.min_pop_size = 4
        self.max_pop_size = 50
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.1, 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        F = np.random.uniform(self.F_min, self.F_max, self.pop_size)
        CR = np.random.uniform(self.CR_min, self.CR_max, self.pop_size)

        while num_evaluations < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    F[i] = np.clip(F[i] + 0.1 * np.random.normal(), self.F_min, self.F_max)
                    CR[i] = np.clip(CR[i] + 0.1 * np.random.normal(), self.CR_min, self.CR_max)
                else:
                    F[i] = np.clip(F[i] - 0.1 * np.random.normal(), self.F_min, self.F_max)
                    CR[i] = np.clip(CR[i] - 0.1 * np.random.normal(), self.CR_min, self.CR_max)

                if num_evaluations >= self.budget:
                    break

            # Dynamic population size adaptation
            if num_evaluations < self.budget and self.pop_size < self.max_pop_size:
                self.pop_size = min(self.pop_size + 1, self.max_pop_size)
                new_individual = np.random.uniform(lb, ub, self.dim)
                population = np.vstack([population, new_individual])
                fitness = np.append(fitness, func(new_individual))
                F = np.append(F, np.random.uniform(self.F_min, self.F_max))
                CR = np.append(CR, np.random.uniform(self.CR_min, self.CR_max))
                num_evaluations += 1

            elif num_evaluations < self.budget and self.pop_size > self.min_pop_size:
                worst_idx = np.argmax(fitness)
                population = np.delete(population, worst_idx, axis=0)
                fitness = np.delete(fitness, worst_idx)
                F = np.delete(F, worst_idx)
                CR = np.delete(CR, worst_idx)
                self.pop_size -= 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]