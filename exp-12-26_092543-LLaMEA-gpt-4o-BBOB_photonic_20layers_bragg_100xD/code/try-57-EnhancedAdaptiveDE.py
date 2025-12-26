import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.pop = None
        self.bounds = None
        self.evaluations = 0

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        lb, ub = self.bounds
        self.pop = lb + (ub - lb) * np.random.rand(self.initial_population_size, self.dim)
        fitness = np.array([func(ind) for ind in self.pop])
        self.evaluations = self.initial_population_size
        successful_mutations = []
        stagnation_counter = 0

        while self.evaluations < self.budget:
            new_pop = np.empty_like(self.pop)
            for i in range(self.pop.shape[0]):
                indices = np.random.choice([idx for idx in range(self.pop.shape[0]) if idx != i], 3, replace=False)
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
                self.evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    successful_mutations.append((F_local, CR_local))
                    stagnation_counter = 0
                else:
                    new_pop[i] = self.pop[i]
                    stagnation_counter += 1

            self.pop = new_pop

            if successful_mutations:
                self.F = np.mean([x[0] for x in successful_mutations])
                self.CR = np.mean([x[1] for x in successful_mutations])
                successful_mutations.clear()

            if stagnation_counter > 100:
                self._diversity_restart(lb, ub, fitness)
                stagnation_counter = 0

            if self.evaluations < self.budget and np.random.rand() < 0.05:
                self._intensify_local_search(func, lb, ub, fitness)

        best_index = np.argmin(fitness)
        return self.pop[best_index]

    def _diversity_restart(self, lb, ub, fitness):
        indices = np.random.choice(self.pop.shape[0], size=int(0.2 * self.pop.shape[0]), replace=False)
        self.pop[indices] = lb + (ub - lb) * np.random.rand(len(indices), self.dim)
        for idx in indices:
            fitness[idx] = func(self.pop[idx])
            self.evaluations += 1

    def _intensify_local_search(self, func, lb, ub, fitness):
        best_idx = np.argmin(fitness)
        for _ in range(5):
            perturbation = 0.1 * (ub - lb) * np.random.randn(self.dim)
            candidate = np.clip(self.pop[best_idx] + perturbation, lb, ub)
            candidate_fitness = func(candidate)
            self.evaluations += 1

            if candidate_fitness < fitness[best_idx]:
                self.pop[best_idx] = candidate
                fitness[best_idx] = candidate_fitness