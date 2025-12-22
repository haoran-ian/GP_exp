import numpy as np

class AdaptiveEnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.8
        self.CR = 0.9
        self.initial_population_size = 10 * self.dim
        self.main_population_size = self.initial_population_size // 2
        self.aux_population_size = self.initial_population_size // 2
        self.main_population = None
        self.aux_population = None
        self.main_fitness = None
        self.aux_fitness = None
        self.memory_archive = []
        self.stagnation_counter = np.zeros(self.main_population_size)
        self.max_stagnation = 10

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.main_population = np.random.uniform(lb, ub, (self.main_population_size, self.dim))
        self.main_fitness = np.full(self.main_population_size, np.inf)
        self.aux_population = np.random.uniform(lb, ub, (self.aux_population_size, self.dim))
        self.aux_fitness = np.full(self.aux_population_size, np.inf)

    def mutate(self, main_idx, aux=False):
        population = self.aux_population if aux else self.main_population
        idxs = [i for i in range(population.shape[0]) if i != main_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate):
        perturbation_strength = 0.1 * (np.random.rand(self.dim) - 0.5)
        return candidate + perturbation_strength

    def adjust_parameters(self, evaluation_ratio):
        self.F = 0.5 + 0.3 * np.sin(np.pi * evaluation_ratio)
        self.CR = 0.9 - 0.5 * evaluation_ratio

    def update_memory_archive(self, candidate, candidate_fitness):
        self.memory_archive.append((candidate, candidate_fitness))
        if len(self.memory_archive) > self.main_population_size:
            self.memory_archive.pop(0)

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)

            for i in range(self.main_population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.mutate(i)
                trial = self.crossover(self.main_population[i], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.main_fitness[i]:
                    self.main_population[i] = trial
                    self.main_fitness[i] = trial_fitness
                    self.stagnation_counter[i] = 0
                else:
                    self.stagnation_counter[i] += 1

                if np.random.rand() < 0.3:
                    candidate = self.local_search(self.main_population[i])
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)

                    candidate_fitness = func(candidate)
                    evaluations += 1

                    if candidate_fitness < self.main_fitness[i]:
                        self.main_population[i] = candidate
                        self.main_fitness[i] = candidate_fitness
                        self.stagnation_counter[i] = 0

                self.update_memory_archive(self.main_population[i], self.main_fitness[i])

                if self.stagnation_counter[i] >= self.max_stagnation:
                    lb, ub = func.bounds.lb, func.bounds.ub
                    self.main_population[i] = np.random.uniform(lb, ub, self.dim)
                    self.main_fitness[i] = func(self.main_population[i])
                    evaluations += 1
                    self.stagnation_counter[i] = 0
            
            if evaluations >= self.budget:
                break

            # Auxiliary population evolution for additional exploration
            for j in range(self.aux_population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.mutate(j, aux=True)
                trial = self.crossover(self.aux_population[j], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.aux_fitness[j]:
                    self.aux_population[j] = trial
                    self.aux_fitness[j] = trial_fitness

        best_idx_main = np.argmin(self.main_fitness)
        best_idx_aux = np.argmin(self.aux_fitness)
        return self.main_population[best_idx_main] if self.main_fitness[best_idx_main] < self.aux_fitness[best_idx_aux] else self.aux_population[best_idx_aux]