import numpy as np
from scipy.optimize import basinhopping
from sklearn.cluster import KMeans
from scipy.stats import levy

class EnhancedHybridDEPSOLevy:
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
        self.f = 0.5
        self.cr = 0.9
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.temperature = 1.0
        self.evaluations = 0

    def _calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def _adaptive_parameters(self):
        sigmoid_adaptation = 1.0 / (1.0 + np.exp(-0.1 * (self.evaluations - self.budget / 2)))
        self.f = 0.4 + 0.5 * sigmoid_adaptation
        self.w = 0.4 + 0.5 * sigmoid_adaptation
        if self.evaluations % 50 == 0:
            self.c1, self.c2 = np.random.uniform(1.5, 2.5, 2)  # Randomly adjust cognitive and social factors

    def _local_search(self, func, x0):
        result = basinhopping(lambda x: func(x), x0, minimizer_kwargs={"method": "Nelder-Mead"}, niter=5)
        return result.x, result.fun

    def _levy_flight(self, scale=1.0):
        return levy.rvs(size=self.dim) * scale

    def _dynamic_clustering(self):
        num_clusters = max(2, self.population_size // 10)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(self.population)
        return kmeans.labels_

    def _cooperative_subpopulations(self, func):
        num_subpopulations = self.population_size // 5
        for _ in range(num_subpopulations):
            indices = np.random.choice(self.population_size, 5, replace=False)
            subpopulation = self.population[indices]
            sub_best_idx = np.argmin(self.personal_best_scores[indices])
            sub_best = subpopulation[sub_best_idx]
            sub_best_score = self.personal_best_scores[indices[sub_best_idx]]
            for i in indices:
                if self.evaluations >= self.budget:
                    break
                step = self._levy_flight(scale=self.temperature)
                trial = np.clip(subpopulation[i] + step, func.bounds.lb, func.bounds.ub)
                score = func(trial)
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = trial
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best = trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

        while self.evaluations < self.budget:
            self._adaptive_parameters()
            cluster_labels = self._dynamic_clustering()
            diversity = self._calculate_diversity()
            levy_scale = np.clip((diversity / self.dim) * self.temperature, 0.1, 1.0)

            for i in range(self.population_size):
                has_levy_flight = i % 2 == 0
                if has_levy_flight:
                    step = self._levy_flight(scale=levy_scale)
                    trial = np.clip(self.population[i] + step, lb, ub)
                else:
                    cluster_idx = np.where(cluster_labels == cluster_labels[i])[0]
                    if len(cluster_idx) > 3:
                        idxs = np.random.choice(cluster_idx, 3, replace=False)
                    else:
                        idxs = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = self.population[idxs]
                    mutant = np.clip(x0 + self.f * (x1 - x2) + self.f * (self.population[i] - x0), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, self.population[i])

                if self.evaluations < self.budget:
                    score = func(trial)
                    self.evaluations += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best[i] = trial
                        if score < self.global_best_score:
                            self.global_best_score = score
                            self.global_best = trial

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
                social = self.c2 * r2 * (self.global_best - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

                if self.evaluations >= self.budget:
                    break

            if self.evaluations < self.budget and self.evaluations % 100 == 0:
                best_idx = np.argmin(self.personal_best_scores)
                x0 = self.personal_best[best_idx]
                new_x, new_score = self._local_search(func, x0)
                self.evaluations += 5  # Increment evaluation count by 5 for local search
                if new_score < self.global_best_score:
                    self.global_best_score = new_score
                    self.global_best = new_x

            self._cooperative_subpopulations(func)

        elite_population_idx = np.argsort(self.personal_best_scores)[:5]  # Select top 5 elite solutions
        self.population[elite_population_idx] = self.global_best + 0.1 * np.random.randn(5, self.dim)  # Inject noise

        return self.global_best