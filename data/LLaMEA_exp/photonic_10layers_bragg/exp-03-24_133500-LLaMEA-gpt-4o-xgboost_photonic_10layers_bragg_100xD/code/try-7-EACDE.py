import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class EACDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_range = (0.4, 1.0)
        self.CR_range = (0.1, 0.9)
        self.memory_size = 5
        self.F_memory = np.random.uniform(*self.F_range, self.memory_size)
        self.CR_memory = np.random.uniform(*self.CR_range, self.memory_size)
        self.memory_index = 0
        self.bounds = None

    def initialize_population(self):
        lower, upper = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lower, upper, (self.population_size, self.dim))

    def select_parents(self, population, scores):
        idx = np.random.choice(self.population_size, 3, replace=False)
        return population[idx], scores[idx]

    def mutate(self, target, best, r1, r2, F):
        return np.clip(best + F * (r1 - r2), self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def adapt_parameters(self, success, F, CR):
        if success:
            self.F_memory[self.memory_index] = F
            self.CR_memory[self.memory_index] = CR
            self.memory_index = (self.memory_index + 1) % self.memory_size
        else:
            F = np.random.choice(self.F_memory)
            CR = np.random.choice(self.CR_memory)
        return F, CR

    def cluster_population(self, population):
        kmeans = KMeans(n_clusters=min(self.population_size // 20, 5), n_init=1)
        kmeans.fit(population)
        return kmeans.labels_

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            clusters = self.cluster_population(population)
            for i in range(self.population_size):
                cluster_indices = np.where(clusters == clusters[i])[0]
                if len(cluster_indices) > 1:
                    target = population[i]
                    best_idx = cluster_indices[np.argmin(scores[cluster_indices])]
                    best = population[best_idx]
                    r1, r2 = population[np.random.choice(cluster_indices, 2, replace=False)]
                    F = np.random.choice(self.F_memory)
                    CR = np.random.choice(self.CR_memory)
                    mutant = self.mutate(target, best, r1, r2, F)
                    trial = self.crossover(target, mutant, CR)
                    trial_score = func(trial)
                    evals += 1
                    if trial_score < scores[i]:
                        population[i] = trial
                        scores[i] = trial_score
                        F, CR = self.adapt_parameters(success=True, F=F, CR=CR)
                    else:
                        F, CR = self.adapt_parameters(success=False, F=F, CR=CR)
                    if evals >= self.budget:
                        break
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]