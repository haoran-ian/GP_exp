import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_mean = 0.5
        self.CR_mean = 0.9
        self.initial_population_size = 10 * self.dim
        self.population_size = self.initial_population_size
        self.population = None
        self.fitness = None
        self.memory_archive = []
        self.stagnation_counter = np.zeros(self.initial_population_size)
        self.max_stagnation = 10
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
        mutant = self.population[a] + self.F_mean * (self.population[b] - self.population[c])
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR_mean
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def local_search(self, candidate):
        perturbation_strength = 0.1 * (np.random.rand(self.dim) - 0.5)
        return candidate + perturbation_strength

    def adjust_parameters(self, evaluation_ratio):
        if len(self.parameter_memory) >= 5:
            self.F_mean = np.mean([p['F'] for p in self.parameter_memory[-5:]])
            self.CR_mean = np.mean([p['CR'] for p in self.parameter_memory[-5:]])
        else:
            self.F_mean = 0.5 + 0.3 * np.sin(np.pi * evaluation_ratio)
            self.CR_mean = 0.9 - 0.5 * evaluation_ratio

    def resize_population(self, evaluations):
        if evaluations > self.budget * self.dynamic_resizing_threshold:
            self.population_size = self.initial_population_size // 2
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]
            self.stagnation_counter = self.stagnation_counter[:self.population_size]

    def update_memory_archive(self, candidate, candidate_fitness, f, cr):
        self.memory_archive.append((candidate, candidate_fitness))
        self.parameter_memory.append({'F': f, 'CR': cr})
        if len(self.memory_archive) > self.population_size:
            self.memory_archive.pop(0)
            self.parameter_memory.pop(0)

    def apply_clustering(self, phase):
        n_clusters = max(2, self.population_size // (10 * phase))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.population)
        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                self.population[i] += 0.05 * (cluster_center - self.population[i])

    def diversify_and_intensify(self, func, candidate, candidate_fitness):
        if np.random.rand() < 0.3:
            candidate = self.local_search(candidate)
            candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
            new_fitness = func(candidate)
            return candidate, new_fitness if new_fitness < candidate_fitness else candidate_fitness
        return candidate, candidate_fitness

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)
            self.resize_population(evaluations)

            phase = 1 if evaluation_ratio < self.clustering_phase_threshold else 2
            if evaluations % int((1 - 0.5 * evaluation_ratio) * self.budget / (10 * phase)) == 0:
                self.apply_clustering(phase)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                f = self.F_mean + 0.1 * (np.random.rand() - 0.5)
                cr = self.CR_mean + 0.1 * (np.random.rand() - 0.5)
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i], self.fitness[i] = trial, trial_fitness
                    self.stagnation_counter[i] = 0
                else:
                    self.stagnation_counter[i] += 1

                candidate, candidate_fitness = self.diversify_and_intensify(func, self.population[i], self.fitness[i])
                if candidate_fitness < self.fitness[i]:
                    self.population[i], self.fitness[i] = candidate, candidate_fitness
                    self.stagnation_counter[i] = 0

                self.update_memory_archive(self.population[i], self.fitness[i], f, cr)

                if self.stagnation_counter[i] >= self.max_stagnation:
                    lb, ub = func.bounds.lb, func.bounds.ub
                    self.population[i] = np.random.uniform(lb, ub, self.dim)
                    self.fitness[i] = func(self.population[i])
                    evaluations += 1
                    self.stagnation_counter[i] = 0

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]