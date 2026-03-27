import numpy as np
from sklearn.cluster import KMeans

class EnhancedHybridPSO_DE_Clustering:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = None
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_score = np.inf
        self.c1_initial = 2.1
        self.c2_initial = 2.0
        self.w_initial = 0.9
        self.w_final = 0.4
        self.mutation_factor = 0.9
        self.elitism_rate = 0.1

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            w = self.w_final + (self.w_initial - self.w_final) * (self.budget - eval_count) / self.budget
            c1 = self.c1_initial - (0.5 * eval_count / self.budget)
            c2 = self.c2_initial + (0.5 * eval_count / self.budget)

            for i in range(self.population_size):
                position = self.particles[i]
                score = func(position)
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = position

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = position

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (w * self.velocities[i] +
                                      c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                      c2 * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds_lb, bounds_ub)

            kmeans = KMeans(n_clusters=min(5, self.population_size // 2), n_init=1)
            kmeans.fit(self.particles)
            cluster_centers = kmeans.cluster_centers_

            for center in cluster_centers:
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                
                mutant_vector = x0 + self.mutation_factor * (x1 - x2)
                mutant_vector = np.clip(mutant_vector, bounds_lb, bounds_ub)

                dynamic_crossover_rate = 0.9 - 0.5 * (eval_count / self.budget)
                crossover_mask = np.random.rand(self.dim) < dynamic_crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, center)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best = trial_vector

            if eval_count >= self.budget:
                break

            top_indices = np.argsort(self.personal_best_scores)[:int(self.elitism_rate * self.population_size)]
            elites = self.personal_best[top_indices]
            self.particles[top_indices] = elites

        return self.global_best