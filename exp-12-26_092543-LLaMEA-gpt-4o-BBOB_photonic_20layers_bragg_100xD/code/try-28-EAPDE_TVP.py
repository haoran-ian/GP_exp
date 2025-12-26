import numpy as np

class EAPDE_TVP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.min_population_size = 5 * dim
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover rate
        self.pop = None
        self.bounds = None

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.initial_population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations = self.initial_population_size
        successful_mutations = []

        while evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(len(self.pop)):
                indices = np.random.choice([idx for idx in range(len(self.pop)) if idx != i], 3, replace=False)
                r1, r2, r3 = self.pop[indices]

                F_local = np.clip(self.F + 0.1 * np.random.randn(), 0, 1)
                CR_local = np.clip(self.CR + 0.05 * np.random.randn(), 0, 1)
                mutant = r1 + F_local * (r2 - r3)
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < CR_local
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    successful_mutations.append((F_local, CR_local))
                else:
                    new_pop[i] = self.pop[i]

            self.pop = new_pop

            if successful_mutations:
                self.F = np.mean([x[0] for x in successful_mutations])
                self.CR = np.mean([x[1] for x in successful_mutations])
                successful_mutations.clear()

            if evaluations < self.budget and np.random.rand() < 0.1:
                rand_index = np.random.randint(len(self.pop))
                self.pop[rand_index] = lb + (ub - lb) * np.random.rand(self.dim)
                fitness[rand_index] = func(self.pop[rand_index])
                evaluations += 1

            if len(self.pop) > self.min_population_size:
                diversity = np.std(fitness)
                if diversity < 1e-5:
                    removal_count = max(1, int(0.05 * len(self.pop)))
                    worst_indices = np.argsort(fitness)[-removal_count:]
                    self.pop = np.delete(self.pop, worst_indices, axis=0)
                    fitness = np.delete(fitness, worst_indices)

        best_index = np.argmin(fitness)
        return self.pop[best_index]