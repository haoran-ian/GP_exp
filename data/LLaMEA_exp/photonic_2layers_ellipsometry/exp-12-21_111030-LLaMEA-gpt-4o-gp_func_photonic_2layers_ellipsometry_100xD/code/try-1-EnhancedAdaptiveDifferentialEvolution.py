import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.population_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.elite = None

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def dynamic_parameters(self, evaluations):
        self.F = 0.5 + (0.3 * np.sin(np.pi * evaluations / self.budget))
        self.CR = 0.8 + (0.2 * np.cos(np.pi * evaluations / self.budget))

    def mutate(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        return self.population[a] + self.F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def local_search(self, candidate):
        perturbation_strength = 0.05 * (np.random.rand(self.dim) - 0.5)
        return candidate + perturbation_strength

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            self.dynamic_parameters(evaluations)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                # Perform adaptive local search
                if np.random.rand() < 0.3:  # 30% chance to apply local search
                    candidate = self.local_search(self.population[i])
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)

                    candidate_fitness = func(candidate)
                    evaluations += 1

                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate
                        self.fitness[i] = candidate_fitness

            # Update the elite
            best_idx = np.argmin(self.fitness)
            if self.elite is None or self.fitness[best_idx] < func(self.elite):
                self.elite = self.population[best_idx].copy()

        return self.elite