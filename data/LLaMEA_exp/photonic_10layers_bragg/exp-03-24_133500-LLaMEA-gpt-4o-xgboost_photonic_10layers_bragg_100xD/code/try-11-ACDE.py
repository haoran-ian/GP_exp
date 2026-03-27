import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class ACDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.F_range = (0.3, 0.9)
        self.CR_range = (0.2, 0.8)
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

    def adapt_parameters(self, F, CR, success=False):
        if success:
            F = min(self.F_range[1], max(self.F_range[0], F + 0.02 * (self.F_range[1] - F)))
            CR = min(self.CR_range[1], max(self.CR_range[0], CR + 0.02 * (self.CR_range[1] - CR)))
        else:
            F = max(self.F_range[0], F - 0.02 * (F - self.F_range[0]))
            CR = max(self.CR_range[0], CR - 0.02 * (CR - self.CR_range[0]))
        return F, CR

    def dynamic_cluster_population(self, population):
        kmeans = KMeans(n_clusters=max(2, min(self.population_size // 30, 10)), n_init=1)
        kmeans.fit(population)
        return kmeans.labels_

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        F, CR = np.random.uniform(*self.F_range), np.random.uniform(*self.CR_range)
        
        evals = self.population_size
        while evals < self.budget:
            clusters = self.dynamic_cluster_population(population)
            for i in range(self.population_size):
                cluster_indices = np.where(clusters == clusters[i])[0]
                if len(cluster_indices) > 2:
                    target = population[i]
                    best_idx = cluster_indices[np.argmin(scores[cluster_indices])]
                    best = population[best_idx]
                    r1, r2 = population[np.random.choice(cluster_indices, 2, replace=False)]
                    r3 = population[np.random.choice(np.setdiff1d(np.arange(self.population_size), cluster_indices), 1)[0]]
                    F1, F2 = np.random.uniform(*self.F_range, 2)
                    mutant = np.clip(best + F1 * (r1 - r2) + F2 * (r2 - r3), self.bounds.lb, self.bounds.ub)
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
            if evals % (self.budget // 5) == 0:
                self.population_size = max(self.initial_population_size // 2, self.population_size - dim)
                population = population[np.argsort(scores)[:self.population_size]]
                scores = scores[np.argsort(scores)[:self.population_size]]
        
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]