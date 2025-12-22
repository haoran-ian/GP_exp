import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveClusteringDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_base = 0.5
        self.CR_base = 0.7
        self.initial_population_size = 10 * self.dim
        self.population_size = self.initial_population_size
        self.population = None
        self.fitness = None
        self.memory_archive = []
        self.stagnation_counter = np.zeros(self.initial_population_size)
        self.max_stagnation = 10
        self.elite_size = max(1, self.population_size // 10)
        self.dynamic_resizing_threshold = 0.5
        self.clustering_phase_threshold = 0.3
        self.parameter_memory = []

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
        perturbation_strength = 0.05 * (np.random.rand(self.dim) - 0.5)
        return candidate + perturbation_strength

    def adjust_parameters(self, evaluation_ratio):
        if self.parameter_memory:
            self.F = np.mean([p['F'] for p in self.parameter_memory[-5:]])
            self.CR = np.mean([p['CR'] for p in self.parameter_memory[-5:]])
        else:
            self.F = self.F_base + 0.3 * np.sin(np.pi * evaluation_ratio)
            self.CR = self.CR_base - 0.5 * evaluation_ratio

    def resize_population(self, evaluations):
        if evaluations > self.budget * self.dynamic_resizing_threshold:
            self.population_size = self.initial_population_size // 2
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]
            self.stagnation_counter = self.stagnation_counter[:self.population_size]

    def update_memory_archive(self, candidate, candidate_fitness):
        if len(self.memory_archive) < self.population_size:
            self.memory_archive.append((candidate, candidate_fitness))
        else:
            worst_idx = np.argmax([fit for _, fit in self.memory_archive])
            if candidate_fitness < self.memory_archive[worst_idx][1]:
                self.memory_archive[worst_idx] = (candidate, candidate_fitness)

    def apply_clustering(self, phase):
        n_clusters = max(2, self.population_size // (10 * phase))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.population)
        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                self.population[i] += 0.05 * (cluster_center - self.population[i])

    def preserve_elites(self):
        elite_indices = np.argsort(self.fitness)[:self.elite_size]
        for idx in elite_indices:
            self.population[idx] = self.memory_archive[idx % len(self.memory_archive)][0]

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)
            self.resize_population(evaluations)

            if evaluation_ratio < self.clustering_phase_threshold:
                phase = 1
            else:
                phase = 2

            if evaluations % int((1 - 0.5 * evaluation_ratio) * self.budget / (10 * phase)) == 0:
                self.apply_clustering(phase)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                f = self.F + 0.1 * (np.random.rand() - 0.5)
                cr = self.CR + 0.1 * (np.random.rand() - 0.5)

                mutant = self.mutate(i)
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

            self.preserve_elites()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]