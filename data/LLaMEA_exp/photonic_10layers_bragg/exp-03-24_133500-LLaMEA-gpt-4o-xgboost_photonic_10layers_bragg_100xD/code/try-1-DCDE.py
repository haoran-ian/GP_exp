import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class DCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.max_population_size = 20 * dim
        self.F_range = (0.5, 1.0)
        self.CR_range = (0.1, 0.9)
        self.bounds = None

    def initialize_population(self, size):
        lower, upper = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lower, upper, (size, self.dim))

    def select_parents(self, population, scores):
        idx = np.random.choice(len(population), 3, replace=False)
        return population[idx], scores[idx]

    def mutate(self, target, best, r1, r2, F):
        return np.clip(best + F * (r1 - r2), self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def adapt_parameters(self, F, CR, success=False):
        if success:
            F = min(self.F_range[1], max(self.F_range[0], F + 0.1 * (self.F_range[1] - F)))
            CR = min(self.CR_range[1], max(self.CR_range[0], CR + 0.1 * (self.CR_range[1] - CR)))
        else:
            F = max(self.F_range[0], F - 0.1 * (F - self.F_range[0]))
            CR = max(self.CR_range[0], CR - 0.1 * (CR - self.CR_range[0]))
        return F, CR

    def dynamic_cluster_population(self, population):
        fitness_diversity = np.std([func(ind) for ind in population])
        n_clusters = max(2, min(len(population) // 20, 5 + int(fitness_diversity * 10)))
        kmeans = KMeans(n_clusters=n_clusters, n_init=1)
        kmeans.fit(population)
        return kmeans.labels_

    def __call__(self, func):
        self.bounds = func.bounds
        population_size = self.initial_population_size
        population = self.initialize_population(population_size)
        scores = np.array([func(ind) for ind in population])
        F, CR = np.random.uniform(*self.F_range), np.random.uniform(*self.CR_range)

        evals = population_size
        while evals < self.budget:
            clusters = self.dynamic_cluster_population(population)
            for i in range(len(population)):
                cluster_indices = np.where(clusters == clusters[i])[0]
                if len(cluster_indices) > 1:
                    target = population[i]
                    best_idx = cluster_indices[np.argmin(scores[cluster_indices])]
                    best = population[best_idx]
                    r1, r2 = population[np.random.choice(cluster_indices, 2, replace=False)]
                    mutant = self.mutate(target, best, r1, r2, F)
                    trial = self.crossover(target, mutant, CR)
                    trial_score = func(trial)
                    evals += 1
                    if trial_score < scores[i]:
                        population[i] = trial
                        scores[i] = trial_score
                        F, CR = self.adapt_parameters(F, CR, success=True)
                    else:
                        F, CR = self.adapt_parameters(F, CR, success=False)
                    if evals >= self.budget:
                        break

            # Dynamically adjust population size
            if evals < self.budget:
                diversity = np.std(scores)
                if diversity < 0.01 and population_size < self.max_population_size:
                    new_individuals = self.initialize_population(self.dim)
                    population = np.vstack((population, new_individuals))
                    new_scores = np.array([func(ind) for ind in new_individuals])
                    scores = np.concatenate((scores, new_scores))
                    population_size += self.dim
                    evals += self.dim

        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]