import numpy as np

class ImprovedHybridDifferentialEvolutionPSO:
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
        self.f = 0.5  # Scaling factor for differential evolution
        self.cr = 0.9  # Crossover probability for differential evolution
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 2.0  # Cognitive (personal) component for PSO
        self.c2 = 2.0  # Social (global) component for PSO

    def _calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            diversity = self._calculate_diversity()
            self.w = 0.1 + 0.4 * (1 - diversity / (np.linalg.norm(ub - lb) / 2))
            self.f = 0.6 + 0.2 * (1 - diversity / (np.linalg.norm(ub - lb) / 2))  # Modified line

            for i in range(self.population_size):
                # Differential evolution mutation and crossover
                while True:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    if i not in idxs:
                        break
                x0, x1, x2 = self.population[idxs]
                mutant = np.clip(x0 + self.f * (x1 - x2), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.population[i])

                # Evaluate trial vector
                score = func(trial)
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = trial
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best = trial

                # Update particle velocity and position using PSO
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
                social = self.c2 * r2 * (self.global_best - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

                # Check budget
                if evaluations >= self.budget:
                    break

        return self.global_best