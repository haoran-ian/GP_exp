import numpy as np

class RefinedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_max, self.c1_min = 2.5, 0.5
        self.c2_max, self.c2_min = 2.5, 0.5
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
        
        while self.evaluations < self.budget:
            global_best_value = func(self.global_best)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Adaptive inertia weight
                self.w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)

                # Dynamic learning factors
                self.c1 = self.c1_min + (self.c1_max - self.c1_min) * (1 - self.evaluations / self.budget)
                self.c2 = self.c2_min + (self.c2_max - self.c2_min) * (self.evaluations / self.budget)

                # Particle Swarm Optimization step
                r1, r2 = np.random.rand(), np.random.rand()
                damping_factor = 0.99 * (1 - self.evaluations / self.budget)  # Introduce decay factor
                self.vel[i] = damping_factor * (self.w * self.vel[i] + 
                               self.c1 * r1 * (self.personal_best[i] - self.pop[i]) + 
                               self.c2 * r2 * (self.global_best - self.pop[i]))
                candidate = self.pop[i] + self.vel[i]
                candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                candidate_value = func(candidate)

                self.evaluations += 1

                if candidate_value < func(self.personal_best[i]):
                    self.personal_best[i] = candidate
                    if candidate_value < func(self.global_best):
                        self.global_best = candidate

                # Hybridized Differential Evolution step with mutation strategies
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
                # Use a linear decreasing factor for F
                f_dynamic = self.f * (1 - self.evaluations / self.budget)
                mutant = np.clip(a + f_dynamic * (b - c), func.bounds.lb, func.bounds.ub)
                # Add a perturbation term to encourage exploration
                perturbation = np.random.normal(0, 0.1, self.dim) * (1 - self.evaluations / self.budget)
                mutant = mutant + perturbation
                trial = np.array([mutant[j] if np.random.rand() < self.cr else self.pop[i][j] for j in range(self.dim)])
                trial_value = func(trial)

                self.evaluations += 1

                if trial_value < candidate_value:
                    self.pop[i] = trial
                    if trial_value < func(self.personal_best[i]):
                        self.personal_best[i] = trial
                        if trial_value < func(self.global_best):
                            self.global_best = trial

        return self.global_best