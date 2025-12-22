import numpy as np

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
        self.cr = 0.9  # Initial crossover probability for differential evolution
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 2.0  # Cognitive (personal) component for PSO
        self.c2 = 2.0  # Social (global) component for PSO
        self.success_rate = 0.0

    def _calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def _adapt_parameters(self):
        # Adjust parameters based on success rate
        if self.success_rate > 0.2:
            self.f *= 1.1
            self.cr *= 0.9
        else:
            self.f *= 0.9
            self.cr *= 1.1

        # Ensure parameters are within bounds
        self.f = np.clip(self.f, 0.1, 0.9)
        self.cr = np.clip(self.cr, 0.1, 0.9)

    def _local_search(self, individual, lb, ub):
        # Perform a simple neighborhood-based local search
        perturbation = np.random.normal(0, 0.1, self.dim)
        candidate = np.clip(individual + perturbation, lb, ub)
        return candidate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

        evaluations = 0
        self.success_rate = 0.0

        while evaluations < self.budget:
            diversity = self._calculate_diversity()
            self.w = 0.1 + 0.4 * (1 - diversity / (np.linalg.norm(ub - lb) / 2))
            self._adapt_parameters()

            successes = 0
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

                # Local search
                if evaluations < self.budget:
                    candidate = self._local_search(trial, lb, ub)
                    candidate_score = func(candidate)
                    evaluations += 1
                    if candidate_score < score:
                        trial, score = candidate, candidate_score

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = trial
                    successes += 1
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

            self.success_rate = successes / self.population_size

        return self.global_best