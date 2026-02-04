import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9  # Inertia weight
        self.f_min = 0.5  # Minimum mutation factor
        self.f_max = 0.9  # Maximum mutation factor
        self.cr = 0.9  # Crossover rate
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

                # Adaptive mutation factor
                self.f = self.f_min + (self.f_max - self.f_min) * (1 - self.evaluations / self.budget)

                # Adjust inertia weight and crossover rate
                self.w = 0.9 - 0.8 * (self.evaluations / self.budget)
                self.cr = 0.9 - 0.3 * (func(self.personal_best[i]) / global_best_value)

                # Particle Swarm Optimization step
                r1, r2 = np.random.rand(), np.random.rand()
                self.vel[i] = (self.w * self.vel[i] + 
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

                # Differential Evolution step
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)
                trial = np.array([mutant[j] if np.random.rand() < self.cr else self.pop[i][j] for j in range(self.dim)])

                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < candidate_value:
                    self.pop[i] = trial
                    if trial_value < func(self.personal_best[i]):
                        self.personal_best[i] = trial
                        if trial_value < func(self.global_best):
                            self.global_best = trial

            # Dynamic swarm size adjustment for better global search
            if self.evaluations % (self.budget // 10) == 0 and self.population_size > 10:
                self.population_size -= 1
                self.pop = np.random.rand(self.population_size, self.dim)

        return self.global_best