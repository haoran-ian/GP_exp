import numpy as np
from sklearn.cluster import KMeans

class AdvancedMultiPopulationAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_base = 0.8
        self.CR_base = 0.9
        self.initial_population_size = 10 * self.dim
        self.num_subpopulations = 3
        self.subpopulations = [None] * self.num_subpopulations
        self.sub_fitness = [None] * self.num_subpopulations
        self.memory_archive = []
        self.stagnation_counters = [np.zeros(self.initial_population_size) for _ in range(self.num_subpopulations)]
        self.max_stagnation = 10
        self.clustering_interval = 0.1 * self.budget
        self.parameter_memory = []

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.num_subpopulations):
            pop_size = self.initial_population_size // self.num_subpopulations
            self.subpopulations[i] = np.random.uniform(lb, ub, (pop_size, self.dim))
            self.sub_fitness[i] = np.full(pop_size, np.inf)

    def mutate(self, sub_idx, idx):
        pop_size = len(self.subpopulations[sub_idx])
        idxs = [i for i in range(pop_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.subpopulations[sub_idx][a] + self.F_base * (self.subpopulations[sub_idx][b] - self.subpopulations[sub_idx][c])
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR_base
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def adjust_parameters(self, evaluation_ratio):
        if self.parameter_memory:
            self.F_base = np.mean([p['F'] for p in self.parameter_memory[-5:]])
            self.CR_base = np.mean([p['CR'] for p in self.parameter_memory[-5:]])
        else:
            self.F_base = 0.5 + 0.3 * np.sin(np.pi * evaluation_ratio)
            self.CR_base = 0.9 - 0.5 * evaluation_ratio

    def update_memory_archive(self, candidate, candidate_fitness, f, cr):
        self.memory_archive.append((candidate, candidate_fitness))
        self.parameter_memory.append({'F': f, 'CR': cr})
        if len(self.memory_archive) > self.initial_population_size:
            self.memory_archive.pop(0)
            self.parameter_memory.pop(0)

    def apply_clustering(self, sub_idx):
        kmeans = KMeans(n_clusters=max(2, len(self.subpopulations[sub_idx]) // 10))
        kmeans.fit(self.subpopulations[sub_idx])
        for i in range(len(self.subpopulations[sub_idx])):
            if np.random.rand() < 0.1:
                cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                self.subpopulations[sub_idx][i] += 0.05 * (cluster_center - self.subpopulations[sub_idx][i])

    def __call__(self, func):
        self.initialize(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            self.adjust_parameters(evaluation_ratio)

            for sub_idx in range(self.num_subpopulations):
                if evaluations % self.clustering_interval == 0:
                    self.apply_clustering(sub_idx)

                for i in range(len(self.subpopulations[sub_idx])):
                    if evaluations >= self.budget:
                        break

                    f = self.F_base + 0.1 * (np.random.rand() - 0.5)
                    cr = self.CR_base + 0.1 * (np.random.rand() - 0.5)

                    mutant = self.mutate(sub_idx, i)
                    trial = self.crossover(self.subpopulations[sub_idx][i], mutant)
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                    trial_fitness = func(trial)
                    evaluations += 1

                    if trial_fitness < self.sub_fitness[sub_idx][i]:
                        self.subpopulations[sub_idx][i] = trial
                        self.sub_fitness[sub_idx][i] = trial_fitness
                        self.stagnation_counters[sub_idx][i] = 0
                    else:
                        self.stagnation_counters[sub_idx][i] += 1

                    self.update_memory_archive(self.subpopulations[sub_idx][i], self.sub_fitness[sub_idx][i], f, cr)

                    if self.stagnation_counters[sub_idx][i] >= self.max_stagnation:
                        lb, ub = func.bounds.lb, func.bounds.ub
                        self.subpopulations[sub_idx][i] = np.random.uniform(lb, ub, self.dim)
                        self.sub_fitness[sub_idx][i] = func(self.subpopulations[sub_idx][i])
                        evaluations += 1
                        self.stagnation_counters[sub_idx][i] = 0

        best_overall_idx = np.argmin([np.min(fitness) for fitness in self.sub_fitness])
        best_idx_in_subpop = np.argmin(self.sub_fitness[best_overall_idx])
        return self.subpopulations[best_overall_idx][best_idx_in_subpop]