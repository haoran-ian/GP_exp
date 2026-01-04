import numpy as np

class AdaptivePSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.c3 = 0.5  # New exploration factor
        self.w_max = 0.9
        self.w_min = 0.4
        self.f = 0.9
        self.cr = 0.9
        self.pop = np.random.rand(self.population_size, self.dim)
        self.vel = np.random.randn(self.population_size, self.dim) * 0.1
        self.personal_best = self.pop.copy()
        self.global_best = self.pop[np.argmin([np.inf] * self.population_size)]
        self.evaluations = 0

    def __call__(self, func):
        self.pop = self.pop * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
        self.personal_best = self.pop.copy()
        self.global_best = self.pop[np.argmin([func(ind) for ind in self.pop])]
        personal_best_values = np.array([func(ind) for ind in self.personal_best])

        while self.evaluations < self.budget:
            global_best_value = func(self.global_best)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Adaptive inertia weight
                self.w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)

                # Dynamic learning factors
                self.c1 = 2.0 - 1.5 * (self.evaluations / self.budget)
                self.c2 = 1.5 + 1.5 * (self.evaluations / self.budget)

                # Particle Swarm Optimization step
                r1, r2 = np.random.rand(), np.random.rand()
                exploration_term = self.c3 * np.random.rand(self.dim) * (np.mean(self.pop, axis=0) - self.pop[i])
                self.vel[i] = (self.w * self.vel[i] + 
                               self.c1 * r1 * (self.personal_best[i] - self.pop[i]) + 
                               self.c2 * r2 * (self.global_best - self.pop[i]) +
                               exploration_term)
                candidate = self.pop[i] + self.vel[i]
                candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                candidate_value = func(candidate)

                self.evaluations += 1

                if candidate_value < personal_best_values[i]:
                    self.personal_best[i] = candidate
                    personal_best_values[i] = candidate_value
                    if candidate_value < global_best_value:
                        self.global_best = candidate
                        global_best_value = candidate_value

                # Differential Evolution step
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)
                trial = np.array([mutant[j] if np.random.rand() < self.cr else self.pop[i][j] for j in range(self.dim)])
                trial_value = func(trial)

                self.evaluations += 1

                if trial_value < candidate_value:
                    self.pop[i] = trial
                    if trial_value < personal_best_values[i]:
                        self.personal_best[i] = trial
                        personal_best_values[i] = trial_value
                        if trial_value < global_best_value:
                            self.global_best = trial
                            global_best_value = trial_value

        return self.global_best