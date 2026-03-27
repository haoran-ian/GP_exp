import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class ACDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_range = (0.3, 1.0)  # Changed 0.4 to 0.3 to further enhance exploration
        self.CR_range = (0.1, 0.9)
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
            F = min(self.F_range[1], max(self.F_range[0], F + 0.15 * (self.F_range[1] - F)))  # Increased adaptation step
            CR = min(self.CR_range[1], max(self.CR_range[0], CR + 0.15 * (self.CR_range[1] - CR)))  # Increased adaptation step
        else:
            F = max(self.F_range[0], F - 0.05 * (F - self.F_range[0]))  # Decreased adaptation step
            CR = max(self.CR_range[0], CR - 0.05 * (CR - self.CR_range[0]))  # Decreased adaptation step
        return F, CR

    def cluster_population(self, population):
        kmeans = KMeans(n_clusters=min(self.population_size // 15, 5), n_init=1)  # Adjusted cluster size
        kmeans.fit(population)
        return kmeans.labels_

    def local_search(self, candidate):
        return candidate + np.random.uniform(-0.1, 0.1, self.dim)  # Added local search mechanism

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        F, CR = np.random.uniform(*self.F_range), np.random.uniform(*self.CR_range)

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
                    mutant = self.mutate(target, best, r1, r2, F)
                    trial = self.crossover(target, mutant, CR)
                    trial = self.local_search(trial)  # Apply local search
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
        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]