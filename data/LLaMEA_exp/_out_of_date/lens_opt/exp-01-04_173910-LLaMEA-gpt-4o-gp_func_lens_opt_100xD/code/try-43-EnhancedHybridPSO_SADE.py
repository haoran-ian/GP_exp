import numpy as np

class EnhancedHybridPSO_SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w_max = 0.9
        self.w_min = 0.4
        self.cr_memory = [0.1, 0.2, 0.5, 0.9]
        self.f_memory = [0.4, 0.6, 0.8, 1.0]
        self.success_rates_cr = np.zeros(len(self.cr_memory))
        self.success_rates_f = np.zeros(len(self.f_memory))
        
    def adapt_parameters(self):
        total_success_cr = np.sum(self.success_rates_cr)
        if total_success_cr > 0:
            self.cr_memory = np.array(self.cr_memory) * (np.array(self.success_rates_cr) / total_success_cr)
        total_success_f = np.sum(self.success_rates_f)
        if total_success_f > 0:
            self.f_memory = np.array(self.f_memory) * (np.array(self.success_rates_f) / total_success_f)
        self.success_rates_cr.fill(0)
        self.success_rates_f.fill(0)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_value = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            # Dynamic inertia weight adjustment
            self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            vel = self.w * vel + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            pop = pop + vel
            pop = np.clip(pop, lb, ub)

            # Evaluate new positions
            new_values = np.array([func(ind) for ind in pop])
            evaluations += self.population_size

            # Update personal and global bests
            improvement = new_values < personal_best_value
            personal_best[improvement] = pop[improvement]
            personal_best_value[improvement] = new_values[improvement]

            if np.min(personal_best_value) < global_best_value:
                global_best = personal_best[np.argmin(personal_best_value)]
                global_best_value = np.min(personal_best_value)

            # Self-Adaptive Differential Evolution (SADE)
            if evaluations < self.budget:
                for i in range(self.population_size):
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                    memory_idx = np.random.choice(len(self.cr_memory))
                    self.cr = self.cr_memory[memory_idx]
                    self.f = self.f_memory[memory_idx]
                    
                    mutant = np.clip(a + self.f * (b - c), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, pop[i])
                    trial_value = func(trial)
                    evaluations += 1

                    # Selection
                    if trial_value < new_values[i]:
                        pop[i] = trial
                        new_values[i] = trial_value
                        # Update memory based on success
                        self.cr_memory[memory_idx] = (self.cr_memory[memory_idx] + self.cr) / 2
                        self.f_memory[memory_idx] = (self.f_memory[memory_idx] + self.f) / 2
                        self.success_rates_cr[memory_idx] += 1
                        self.success_rates_f[memory_idx] += 1
                        if trial_value < personal_best_value[i]:
                            personal_best[i] = trial
                            personal_best_value[i] = trial_value
                            if trial_value < global_best_value:
                                global_best = trial
                                global_best_value = trial_value

            self.adapt_parameters()

        return global_best