import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.population_size = 10 * self.dim
        self.population = None
        self.fitness = None
        self.memory_archive = []

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def mutate(self, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return mutant

    def rotation_invariant_crossover(self, target, mutant):  # Changed
        j_rand = np.random.randint(self.dim)
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]
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
                trial = self.rotation_invariant_crossover(self.population[i], mutant)  # Changed
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
                
                self.update_memory_archive(self.population[i], self.fitness[i])

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]