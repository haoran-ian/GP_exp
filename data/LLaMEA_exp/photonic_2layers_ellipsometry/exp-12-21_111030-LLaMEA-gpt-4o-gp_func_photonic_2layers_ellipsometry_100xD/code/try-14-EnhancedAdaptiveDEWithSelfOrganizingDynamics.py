import numpy as np

class EnhancedAdaptiveDEWithSelfOrganizingDynamics:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.8
        self.CR = 0.9
        self.population_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.memory_archive = []
        self.diversity_threshold = 0.1

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

    def local_search(self, candidate):
        perturbation_strength = 0.1 * (np.random.rand(self.dim) - 0.5)
        return candidate + perturbation_strength

    def adjust_parameters(self, evaluation_ratio):
        self.F = 0.5 + 0.3 * np.sin(np.pi * evaluation_ratio)
        self.CR = 0.9 - 0.5 * evaluation_ratio

    def update_memory_archive(self, candidate, candidate_fitness):
        self.memory_archive.append((candidate, candidate_fitness))
        if len(self.memory_archive) > self.population_size:
            self.memory_archive.pop(0)

    def select_from_memory_archive(self):
        if self.memory_archive:
            return min(self.memory_archive, key=lambda x: x[1])[0]
        return None

    def compute_diversity(self):
        if self.population is not None:
            mean = np.mean(self.population, axis=0)
            diversity = np.mean(np.linalg.norm(self.population - mean, axis=1))
            return diversity
        return 0

    def adjust_population_size(self, diversity):
        if diversity < self.diversity_threshold and self.population_size > 5:
            self.population_size = max(5, int(self.population_size * 0.9))
        elif diversity > 2 * self.diversity_threshold:
            self.population_size = min(100, int(self.population_size * 1.1))

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)

            diversity = self.compute_diversity()
            self.adjust_population_size(diversity)

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
                if np.random.rand() < 0.3:
                    candidate = self.local_search(self.population[i])
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                    
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    
                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate
                        self.fitness[i] = candidate_fitness
                
                self.update_memory_archive(self.population[i], self.fitness[i])

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]