import numpy as np

class HADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.population = None
        self.fitness = None
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.evaluations = 0

    def init_population(self, bounds):
        lower_bound = bounds.lb
        upper_bound = bounds.ub
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def differential_evolution(self, target_idx, bounds):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def local_search(self, cand, bounds):
        perturbation = np.clip(cand + np.random.normal(0, 0.1, self.dim), bounds.lb, bounds.ub)
        return perturbation

    def __call__(self, func):
        self.init_population(func.bounds)
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.population[np.argmin(self.fitness)]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                trial = self.differential_evolution(i, func.bounds)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Apply local search with a probability
                if np.random.rand() < 0.1:
                    local_candidate = self.local_search(self.population[i], func.bounds)
                    local_fitness = func(local_candidate)
                    self.evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness

                if self.evaluations >= self.budget:
                    break

        return self.population[np.argmin(self.fitness)]