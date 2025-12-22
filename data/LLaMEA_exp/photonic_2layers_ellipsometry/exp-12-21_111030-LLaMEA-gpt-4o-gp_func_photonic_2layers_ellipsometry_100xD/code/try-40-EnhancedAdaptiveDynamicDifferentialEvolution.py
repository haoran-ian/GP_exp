import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveDynamicDifferentialEvolution:
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
        self.max_stagnation = 8  # Reduced for quicker adaptability
        self.dynamic_resizing_threshold = 0.5
        self.clustering_interval = 0.2 * self.budget
        self.local_search_prob = 0.4  # Increased probability of local search
        self.memory_archive_size = 5  # Dynamic memory archive

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
        self.memory_archive.sort(key=lambda x: x[1])  # Sort to keep the best
        if len(self.memory_archive) > self.memory_archive_size:
            self.memory_archive.pop()  # Keep only the best solutions

    def select_from_memory_archive(self):
        if self.memory_archive:
            return self.memory_archive[np.random.randint(len(self.memory_archive))][0]
        return None

    def apply_clustering(self):
        kmeans = KMeans(n_clusters=max(2, self.population_size // 10))
        kmeans.fit(self.population)
        for i in range(self.population_size):
            if np.random.rand() < 0.1:  # Randomly perturb some within clusters
                cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                self.population[i] += 0.05 * (cluster_center - self.population[i])

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)
            self.resize_population(evaluations)

            if evaluations % self.clustering_interval == 0:
                self.apply_clustering()

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
                    self.stagnation_counter[i] = 0
                else:
                    self.stagnation_counter[i] += 1

                if np.random.rand() < self.local_search_prob:
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
                    new_candidate = self.select_from_memory_archive()
                    if new_candidate is not None:
                        self.population[i] = new_candidate
                    else:
                        self.population[i] = np.random.uniform(lb, ub, self.dim)
                    self.fitness[i] = func(self.population[i])
                    evaluations += 1
                    self.stagnation_counter[i] = 0

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]