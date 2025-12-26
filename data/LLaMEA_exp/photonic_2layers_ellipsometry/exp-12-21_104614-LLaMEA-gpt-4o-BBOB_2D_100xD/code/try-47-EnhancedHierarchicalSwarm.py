import numpy as np
from scipy.optimize import basinhopping
from sklearn.cluster import KMeans
from scipy.stats import levy

class EnhancedHierarchicalSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.global_population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.global_velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.global_personal_best_positions = np.copy(self.global_population)
        self.global_personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.leader_swarm = np.random.uniform(-5, 5, (self.population_size // 5, self.dim))
        self.leader_velocity = np.random.uniform(-1, 1, (self.population_size // 5, self.dim))
        self.leader_best_positions = np.copy(self.leader_swarm)
        self.leader_best_scores = np.full(self.population_size // 5, np.inf)
        self.leader_global_best_position = None
        self.leader_global_best_score = np.inf
        self.f = 0.5
        self.cr = 0.9
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.temperature = 1.0

    def _adaptive_parameters(self, evaluations):
        self.temperature = max(0.1, 1.0 - evaluations / self.budget)
        sigmoid_adaptation = 1.0 / (1.0 + np.exp(-0.1 * (evaluations - self.budget / 2)))
        self.f = 0.4 + 0.5 * sigmoid_adaptation
        self.w = 0.4 + 0.5 * sigmoid_adaptation

    def _local_search(self, func, x0):
        result = basinhopping(lambda x: func(x), x0, minimizer_kwargs={"method": "Nelder-Mead"}, niter=5)
        return result.x, result.fun

    def _dynamic_clustering(self, population):
        num_clusters = max(2, len(population) // 10)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(population)
        return kmeans.labels_

    def _levy_flight(self, scale=1.0):
        return levy.rvs(size=self.dim) * scale

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0

        while evaluations < self.budget:
            self._adaptive_parameters(evaluations)

            global_cluster_labels = self._dynamic_clustering(self.global_population)
            leader_cluster_labels = self._dynamic_clustering(self.leader_swarm)

            for i in range(self.population_size):
                if np.random.rand() < 0.1:
                    step = self._levy_flight(scale=self.temperature)
                    trial = np.clip(self.global_population[i] + step, lb, ub)
                else:
                    cluster_idx = np.where(global_cluster_labels == global_cluster_labels[i])[0]
                    if len(cluster_idx) > 3:
                        idxs = np.random.choice(cluster_idx, 3, replace=False)
                    else:
                        idxs = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = self.global_population[idxs]
                    mutant = np.clip(x0 + self.f * (x1 - x2) + self.f * (self.global_population[i] - x0), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, self.global_population[i])

                if evaluations < self.budget:
                    score = func(trial)
                    evaluations += 1
                    if score < self.global_personal_best_scores[i]:
                        self.global_personal_best_scores[i] = score
                        self.global_personal_best_positions[i] = trial
                        if score < self.global_best_score:
                            self.global_best_score = score
                            self.global_best_position = trial

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.global_personal_best_positions[i] - self.global_population[i])
                social = self.c2 * r2 * (self.global_best_position - self.global_population[i])
                self.global_velocity[i] = self.w * self.global_velocity[i] + cognitive + social
                self.global_population[i] = np.clip(self.global_population[i] + self.global_velocity[i], lb, ub)

                if evaluations >= self.budget:
                    break

            for j in range(len(self.leader_swarm)):
                if np.random.rand() < 0.2:
                    step = self._levy_flight(scale=self.temperature)
                    trial = np.clip(self.leader_swarm[j] + step, lb, ub)
                else:
                    cluster_idx = np.where(leader_cluster_labels == leader_cluster_labels[j])[0]
                    if len(cluster_idx) > 3:
                        idxs = np.random.choice(cluster_idx, 3, replace=False)
                    else:
                        idxs = np.random.choice(len(self.leader_swarm), 3, replace=False)
                    x0, x1, x2 = self.leader_swarm[idxs]
                    mutant = np.clip(x0 + self.f * (x1 - x2) + self.f * (self.leader_swarm[j] - x0), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, self.leader_swarm[j])

                if evaluations < self.budget:
                    score = func(trial)
                    evaluations += 1
                    if score < self.leader_best_scores[j]:
                        self.leader_best_scores[j] = score
                        self.leader_best_positions[j] = trial
                        if score < self.leader_global_best_score:
                            self.leader_global_best_score = score
                            self.leader_global_best_position = trial

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.leader_best_positions[j] - self.leader_swarm[j])
                social = self.c2 * r2 * (self.leader_global_best_position - self.leader_swarm[j])
                self.leader_velocity[j] = self.w * self.leader_velocity[j] + cognitive + social
                self.leader_swarm[j] = np.clip(self.leader_swarm[j] + self.leader_velocity[j], lb, ub)

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget and evaluations % 100 == 0:
                best_idx = np.argmin(self.global_personal_best_scores)
                x0 = self.global_personal_best_positions[best_idx]
                new_x, new_score = self._local_search(func, x0)
                evaluations += 5
                if new_score < self.global_best_score:
                    self.global_best_score = new_score
                    self.global_best_position = new_x

        return self.global_best_position