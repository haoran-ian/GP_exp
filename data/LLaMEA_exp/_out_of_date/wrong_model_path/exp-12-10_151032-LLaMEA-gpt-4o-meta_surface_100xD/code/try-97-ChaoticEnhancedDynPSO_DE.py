import numpy as np

class ChaoticEnhancedDynPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w_initial = 0.9
        self.w_final = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.F_initial = 0.8
        self.CR_initial = 0.9
        self.evaluations = 0

    def logistic_map(self, x):
        return 4.0 * x * (1 - x)

    def tent_map(self, x):
        return 2 * x if x < 0.5 else 2 * (1 - x)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        self.evaluations += self.population_size
        chaos_param1, chaos_param2 = np.random.rand(), np.random.rand()

        while self.evaluations < self.budget:
            # Chaos-based dynamic parameter adjustment
            chaos_param1 = self.logistic_map(chaos_param1)
            chaos_param2 = self.tent_map(chaos_param2)
            progress_ratio = self.evaluations / self.budget
            w = self.w_initial - (self.w_initial - self.w_final) * progress_ratio
            c1 = self.c1_initial * (1 - progress_ratio) + chaos_param1 * (3.0 - self.c1_initial)
            c2 = self.c2_initial * (1 - progress_ratio) + chaos_param2 * (3.0 - self.c2_initial)
            F = self.F_initial * chaos_param1
            CR = self.CR_initial * chaos_param2
            
            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (w * velocity +
                        c1 * r1 * (personal_best - population) +
                        c2 * r2 * (global_best - population))
            population += velocity
            population = np.clip(population, lb, ub)

            # Differential Evolution update
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                trial_value = func(trial)
                self.evaluations += 1

                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

                if self.evaluations >= self.budget:
                    break

        return global_best