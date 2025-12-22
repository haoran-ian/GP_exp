import numpy as np
from sklearn.cluster import KMeans

class EnhancedDynamicAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.8
        self.CR = 0.9
        self.initial_population_size = 10 * self.dim
        self.population_size = self.initial_population_size
        self.population = None
        self.fitness = None
        self.memory_archive = []
        self.stagnation_counter = np.zeros(self.initial_population_size)
        self.max_stagnation = 10
        self.dynamic_resizing_threshold = 0.5
        self.cluster_interval = 20  # Interval for clustering the population

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

    def resize_population(self, evaluations):
        if evaluations > self.budget * self.dynamic_resizing_threshold:
            self.population_size = self.initial_population_size // 2
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]
            self.stagnation_counter = self.stagnation_counter[:self.population_size]

    def update_memory_archive(self, candidate, candidate_fitness):
        self.memory_archive.append((candidate, candidate_fitness))
        if len(self.memory_archive) > self.population_size:
            self.memory_archive.pop(0)

    def cluster_population(self):
        if self.population_size > 5:
            kmeans = KMeans(n_clusters=max(2, self.population_size // 10), random_state=0)
            kmeans.fit(self.population)
            return kmeans.labels_
        return np.zeros(self.population_size, dtype=int)

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0
        cluster_labels = np.zeros(self.population_size)

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)
            self.resize_population(evaluations)

            if evaluations % self.cluster_interval == 0:
                cluster_labels = self.cluster_population()

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                cluster_idx = cluster_labels[i]
                cluster_indices = np.where(cluster_labels == cluster_idx)[0]
                other_indices = [idx for idx in cluster_indices if idx != i]

                if len(other_indices) >= 3:
                    a, b, c = np.random.choice(other_indices, 3, replace=False)
                else:
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)

                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.stagnation_counter[i] = 0
                else:
                    self.stagnation_counter[i] += 1

                if np.random.rand() < 0.3:
                    candidate = self.local_search(self.population[i])
                    candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)

                    candidate_fitness = func(candidate)
                    evaluations += 1

                    if candidate_fitness < self.fitness[i]:
                        self.population[i] = candidate
                        self.fitness[i] = candidate_fitness
                        self.stagnation_counter[i] = 0

                self.update_memory_archive(self.population[i], self.fitness[i])

                if self.stagnation_counter[i] >= self.max_stagnation:
                    lb, ub = func.bounds.lb, func.bounds.ub
                    self.population[i] = np.random.uniform(lb, ub, self.dim)
                    self.fitness[i] = func(self.population[i])
                    evaluations += 1
                    self.stagnation_counter[i] = 0

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]