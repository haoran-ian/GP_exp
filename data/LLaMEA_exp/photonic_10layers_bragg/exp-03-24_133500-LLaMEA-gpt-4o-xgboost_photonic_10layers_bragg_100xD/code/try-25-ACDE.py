import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class ACDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_range = (0.4, 1.0)
        self.CR_range = (0.1, 0.9)
        self.bounds = None

    def initialize_population(self):
        lower, upper = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lower, upper, (self.population_size, self.dim))

    def select_parents(self, population, scores):
        idx = np.random.choice(self.population_size, 3, replace=False)
        return population[idx], scores[idx]

    def mutate(self, target, elite, r1, r2, F):
        inertia_weight = np.random.uniform(0.5, 0.9)
        return np.clip(target + inertia_weight * (elite + F * (r1 - r2) - target), self.bounds.lb, self.bounds.ub)

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

    def adaptive_clustering(self, population, scores):
        cluster_count = min(max(2, self.population_size // 20), 5)
        kmeans = KMeans(n_clusters=cluster_count, n_init=1)
        kmeans.fit(population)
        return kmeans.labels_

    def elite_learning(self, population, scores, func):
        elite_idx = np.argsort(scores)[:max(1, self.population_size // 10)]
        elite_candidates = population[elite_idx]
        improved_elite = []
        for candidate in elite_candidates:
            noise = np.random.normal(0, 0.1, self.dim)
            new_candidate = np.clip(candidate + noise, self.bounds.lb, self.bounds.ub)
            new_score = func(new_candidate)
            if new_score < scores[elite_idx[0]]:
                improved_elite.append((new_candidate, new_score))
        if improved_elite:
            best_candidate, best_score = min(improved_elite, key=lambda x: x[1])
            best_idx = elite_idx[0]
            population[best_idx] = best_candidate
            scores[best_idx] = best_score

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        F, CR = np.random.uniform(*self.F_range), np.random.uniform(*self.CR_range)

        evals = self.population_size
        while evals < self.budget:
            clusters = self.adaptive_clustering(population, scores)
            for i in range(self.population_size):
                cluster_indices = np.where(clusters == clusters[i])[0]
                if len(cluster_indices) > 1:
                    target = population[i]
                    elite_idx = cluster_indices[np.argmin(scores[cluster_indices])]
                    elite = population[elite_idx]
                    r1, r2 = population[np.random.choice(cluster_indices, 2, replace=False)]
                    mutant = self.mutate(target, elite, r1, r2, F)
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
            self.elite_learning(population, scores, func)
            evals += len(np.unique(clusters))  # Account for elite learning evaluations

        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]