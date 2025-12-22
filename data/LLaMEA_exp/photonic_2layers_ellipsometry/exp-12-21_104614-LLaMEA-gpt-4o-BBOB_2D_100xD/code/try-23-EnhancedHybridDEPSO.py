import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_scores = None
        self.global_best = None
        self.global_best_score = np.inf
        self.f = 0.5  # Initial scaling factor for differential evolution
        self.cr = 0.9  # Crossover probability for differential evolution
        self.w = 0.9  # Initial inertia weight for PSO
        self.c1 = 2.0  # Cognitive (personal) component for PSO
        self.c2 = 2.0  # Social (global) component for PSO
        self.prev_global_best_scores = []

    def _calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def _adaptive_parameters(self, evaluations):
        if len(self.prev_global_best_scores) > 5:
            recent_improvements = np.diff(self.prev_global_best_scores[-5:])
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement < 1e-6:  # If improvement is slow
                self.f = max(0.4, self.f * 0.95)
                self.w = min(0.8, self.w * 1.05)
                self.cr = min(0.95, self.cr * 1.05)  # Adaptive crossover probability
            else:
                self.f = min(0.9, self.f * 1.05)
                self.w = max(0.4, self.w * 0.95)
                self.cr = max(0.7, self.cr * 0.95)  # Adaptive crossover probability

    def _local_search(self, func, x0):
        result = minimize(func, x0, method='Nelder-Mead', options={'maxiter': 50})
        return result.x, result.fun

    def _apply_clustering(self):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(self.population)
        return kmeans.labels_

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            diversity = self._calculate_diversity()
            self._adaptive_parameters(evaluations)

            cluster_labels = self._apply_clustering()

            for i in range(self.population_size):
                # Differential evolution mutation with clustering and Gaussian noise
                cluster_idx = np.where(cluster_labels == cluster_labels[i])[0]
                if len(cluster_idx) > 3:
                    idxs = np.random.choice(cluster_idx, 3, replace=False)
                else:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.population[idxs]
                mutant = np.clip(x0 + self.f * (x1 - x2) + np.random.normal(0, 0.1, self.dim), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.population[i])

                # Evaluate trial vector
                if evaluations < self.budget - 50:
                    score = func(trial)
                    evaluations += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best[i] = trial
                        if score < self.global_best_score:
                            self.global_best_score = score
                            self.global_best = trial
                            self.prev_global_best_scores.append(self.global_best_score)

                # Update particle velocity and position using PSO
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
                social = self.c2 * r2 * (self.global_best - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

                # Check budget
                if evaluations >= self.budget:
                    break

            if evaluations < self.budget and evaluations % 50 == 0:
                # Apply local search
                best_idx = np.argmin(self.personal_best_scores)
                x0 = self.personal_best[best_idx]
                new_x, new_score = self._local_search(func, x0)
                evaluations += 50
                if new_score < self.global_best_score:
                    self.global_best_score = new_score
                    self.global_best = new_x

        return self.global_best