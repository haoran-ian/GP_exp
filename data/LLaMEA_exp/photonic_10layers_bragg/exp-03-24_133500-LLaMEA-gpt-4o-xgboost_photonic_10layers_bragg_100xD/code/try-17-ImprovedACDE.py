import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class ImprovedACDE:
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

    def mutate(self, target, best, r1, r2, F):
        direction = np.random.choice([-1, 1])  # Random direction for exploration
        return np.clip(best + direction * F * (r1 - r2), self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def adapt_parameters(self, F, CR, success=False):
        if success:
            F = min(self.F_range[1], max(self.F_range[0], F + 0.05 * (self.F_range[1] - F)))
            CR = min(self.CR_range[1], max(self.CR_range[0], CR + 0.05 * (self.CR_range[1] - CR)))
        else:
            F = max(self.F_range[0], F - 0.05 * (F - self.F_range[0]))
            CR = max(self.CR_range[0], CR - 0.05 * (CR - self.CR_range[0]))
        return F, CR

    def cluster_population(self, population, scores):
        n_clusters = min(self.population_size // 20, 5)
        kmeans = KMeans(n_clusters=n_clusters, n_init=1)
        kmeans.fit(population)
        dist_to_centroid = cdist(population, kmeans.cluster_centers_, 'euclidean')
        cluster_scores = [scores[np.where(kmeans.labels_ == i)].min() for i in range(n_clusters)]
        best_cluster = np.argmin(cluster_scores)
        return kmeans.labels_, dist_to_centroid[:, best_cluster]

    def dynamic_population_resizing(self, population, scores, phase):
        if phase == 'exploration':
            return population, scores
        else:
            top_k = max(3 * self.dim, self.population_size // 2)
            best_indices = np.argsort(scores)[:top_k]
            return population[best_indices], scores[best_indices]

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        F, CR = np.random.uniform(*self.F_range), np.random.uniform(*self.CR_range)

        evals = self.population_size
        phase = 'exploration'
        while evals < self.budget:
            clusters, distances = self.cluster_population(population, scores)
            for i in range(self.population_size):
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

            population, scores = self.dynamic_population_resizing(population, scores, phase)
            if np.mean(distances) < 0.1 * np.linalg.norm(self.bounds.ub - self.bounds.lb):
                phase = 'exploitation'

        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]