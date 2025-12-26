import numpy as np

class RefinedHybridDifferentialEvolutionPSO:
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
        self.c1 = 1.5  # Cognitive (personal) component for PSO
        self.c2 = 1.5  # Social (global) component for PSO
        self.param_adapt_rate = 0.1  # Rate at which parameters are adapted

    def adapt_parameters(self):
        self.f = np.clip(self.f + self.param_adapt_rate * np.random.uniform(-0.1, 0.1), 0.1, 0.9)
        self.cr = np.clip(self.cr + self.param_adapt_rate * np.random.uniform(-0.1, 0.1), 0.1, 1.0)
        self.w = np.clip(self.w + self.param_adapt_rate * np.random.uniform(-0.1, 0.1), 0.1, 0.9)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
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

            # Adapt parameters periodically
            if evaluations % (self.population_size * 10) == 0:
                self.adapt_parameters()

        return self.global_best