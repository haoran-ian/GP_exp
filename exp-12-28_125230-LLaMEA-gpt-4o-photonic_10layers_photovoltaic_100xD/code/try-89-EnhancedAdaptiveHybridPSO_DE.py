import numpy as np

class EnhancedAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.f_max = 0.9
        self.f_min = 0.5
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
        global_best_value = func(self.global_best)

        # Initialize chaotic sequence for random numbers
        chaos_sequence = np.random.rand(self.population_size, self.dim)
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Adaptive inertia weight
                self.w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)

                # Chaotic sequence for randomness
                chaos_sequence[i] = np.mod(chaos_sequence[i] * (1 - chaos_sequence[i]), 1)

                # Dynamic neighborhood topology
                neighborhood_size = max(1, int(self.population_size * (1 - self.evaluations / self.budget)))
                neighbors = np.random.choice(self.population_size, neighborhood_size, replace=False)
                local_best = self.pop[neighbors[np.argmin([func(self.pop[n]) for n in neighbors])]]

                # Particle Swarm Optimization step
                r1, r2 = chaos_sequence[i], chaos_sequence[i]
                damping_factor = 0.99 * (1 - self.evaluations / self.budget)
                self.vel[i] = damping_factor * (self.w * self.vel[i] + 
                               self.c1 * r1 * (self.personal_best[i] - self.pop[i]) + 
                               self.c2 * r2 * (local_best - self.pop[i]))
                candidate = self.pop[i] + self.vel[i]
                candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                candidate_value = func(candidate)

                self.evaluations += 1

                if candidate_value < func(self.personal_best[i]):
                    self.personal_best[i] = candidate
                    if candidate_value < global_best_value:
                        self.global_best = candidate
                        global_best_value = candidate_value

                # Differential Evolution step
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
                f_dynamic = self.f_min + (self.f_max - self.f_min) * (1 - self.evaluations / self.budget)
                mutant = np.clip(a + f_dynamic * (b - c), func.bounds.lb, func.bounds.ub)
                trial = np.array([mutant[j] if np.random.rand() < self.cr else self.pop[i][j] for j in range(self.dim)])
                trial_value = func(trial)

                self.evaluations += 1

                if trial_value < candidate_value:
                    self.pop[i] = trial
                    if trial_value < func(self.personal_best[i]):
                        self.personal_best[i] = trial
                        if trial_value < global_best_value:
                            self.global_best = trial
                            global_best_value = trial_value

        return self.global_best