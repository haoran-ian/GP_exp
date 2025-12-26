import numpy as np

class AdvancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.memory_archive = []
        self.init_parameters()

    def init_parameters(self):
        self.F = 0.5  # Differential weight initialized
        self.CR = 0.9  # Crossover probability initialized
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.1, 1.0

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def mutate(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def stochastic_local_search(self, candidate):
        perturbation_strength = np.random.laplace(0, 0.1, self.dim)
        return candidate + perturbation_strength

    def adjust_parameters(self, evaluation_ratio):
        self.F = self.F_min + (self.F_max - self.F_min) * np.random.beta(2, 5)
        self.CR = self.CR_min + (self.CR_max - self.CR_min) * evaluation_ratio

    def update_memory_archive(self, candidate, candidate_fitness):
        self.memory_archive.append((candidate, candidate_fitness))
        if len(self.memory_archive) > self.population_size:
            self.memory_archive.pop(0)

    def select_from_memory_archive(self):
        if self.memory_archive:
            return min(self.memory_archive, key=lambda x: x[1])[0]
        return None

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)

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

                if np.random.rand() < 0.3:
                    candidate = self.stochastic_local_search(self.population[i])
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                    
                    candidate_fitness = func(candidate)
                    evaluations += 1

                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate
                        self.fitness[i] = candidate_fitness

                self.update_memory_archive(self.population[i], self.fitness[i])

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]