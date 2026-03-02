import numpy as np

class Hybrid_DE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.F = 0.7  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.c1 = 2.0  # PSO cognitive coefficient
        self.c2 = 2.0  # PSO social coefficient
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.vel_limit = 0.1  # Velocity limit factor
        self.local_search_prob = 0.1  # Probability of local search
        self.adaptive_factor = 0.1  # Factor for controlling adaptivity
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        vel = np.zeros((self.pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.pop_size

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            # DE mutation and crossover
            for i in range(self.pop_size):
                indices = list(range(i)) + list(range(i + 1, self.pop_size))
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                trial_value = func(trial)
                evaluations += 1

                if trial_value < personal_best_values[i]:
                    personal_best[i], personal_best_values[i] = trial, trial_value
                    if trial_value < global_best_value:
                        global_best, global_best_value = trial, trial_value

            # PSO velocity and position update
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = w * vel[i] + self.c1 * r1 * (personal_best[i] - pop[i]) + self.c2 * r2 * (global_best - pop[i])
                vel[i] = np.clip(vel[i], -self.vel_limit * (ub - lb), self.vel_limit * (ub - lb))
                pop[i] = np.clip(pop[i] + vel[i], lb, ub)

                new_value = func(pop[i])
                evaluations += 1
                if new_value < personal_best_values[i]:
                    personal_best[i], personal_best_values[i] = pop[i], new_value
                    if new_value < global_best_value:
                        global_best, global_best_value = pop[i], new_value

                # Local search intensification
                if np.random.rand() < self.local_search_prob:
                    new_trial = pop[i] + self.adaptive_factor * np.random.normal(0, 1, self.dim)
                    new_trial = np.clip(new_trial, lb, ub)
                    local_value = func(new_trial)
                    evaluations += 1
                    if local_value < personal_best_values[i]:
                        personal_best[i], personal_best_values[i] = new_trial, local_value
                        if local_value < global_best_value:
                            global_best, global_best_value = new_trial, local_value
                        
        return global_best