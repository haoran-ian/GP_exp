import numpy as np

class EnhancedAdaptiveDifferentialEvolutionV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.population_size = self.initial_population_size
        self.population = None
        self.fitness = None
        self.memory_archive = []
        self.stagnation_counter = 0
        self.max_stagnation = 10
        self.dynamic_resizing_threshold = 0.5  # Population resize threshold

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def mutate(self, idx, evaluation_ratio):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        F_dynamic = 0.5 + 0.3 * np.sin(np.pi * evaluation_ratio)
        mutant = self.population[a] + F_dynamic * (self.population[b] - self.population[c])
        return mutant

    def crossover(self, target, mutant):
        CR_dynamic = 0.7 + 0.2 * np.random.rand()
        crossover_mask = np.random.rand(self.dim) < CR_dynamic
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate):
        perturbation_strength = 0.1 * (np.random.rand(self.dim) - 0.5)
        return candidate + perturbation_strength

    def resize_population(self, evaluations):
        if evaluations > self.budget * self.dynamic_resizing_threshold:
            self.population_size = max(5, int(self.initial_population_size * (0.5 + 0.5 * (1 - evaluations / self.budget))))
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]

    def update_memory_archive(self, candidate, candidate_fitness):
        self.memory_archive.append((candidate, candidate_fitness))
        if len(self.memory_archive) > self.initial_population_size:
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
            self.resize_population(evaluations)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant = self.mutate(i, evaluation_ratio)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1

                if np.random.rand() < 0.3:
                    candidate = self.local_search(self.population[i])
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)

                    candidate_fitness = func(candidate)
                    evaluations += 1

                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate
                        self.fitness[i] = candidate_fitness
                        self.stagnation_counter = 0

                self.update_memory_archive(self.population[i], self.fitness[i])

                if self.stagnation_counter >= self.max_stagnation:
                    lb, ub = func.bounds.lb, func.bounds.ub
                    self.population[i] = np.random.uniform(lb, ub, self.dim)
                    self.fitness[i] = func(self.population[i])
                    evaluations += 1
                    self.stagnation_counter = 0

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]