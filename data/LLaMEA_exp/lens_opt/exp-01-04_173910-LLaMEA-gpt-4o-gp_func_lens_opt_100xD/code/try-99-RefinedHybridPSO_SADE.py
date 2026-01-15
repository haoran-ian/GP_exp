import numpy as np

class RefinedHybridPSO_SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.6  # cognitive component
        self.c2 = 1.4  # social component
        self.w = 0.6  # inertia weight, slightly decreased for faster convergence
        self.cr_memory = [0.1, 0.3, 0.5, 0.9]
        self.f_memory = [0.3, 0.5, 0.7, 0.9]
        self.success_rate = 0.2  # initial success rate for parameter adaptation

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
            successes = 0
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                memory_idx = np.random.choice(len(self.cr_memory))
                cr = self.cr_memory[memory_idx]
                f = self.f_memory[memory_idx]

                mutant = np.clip(a + f * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < cr
                trial = np.where(crossover, mutant, pop[i])
                trial_value = func(trial)
                evaluations += 1

                # Selection
                if trial_value < new_values[i]:
                    pop[i] = trial
                    new_values[i] = trial_value
                    successes += 1
                    # Update memory based on success
                    self.cr_memory[memory_idx] = (self.cr_memory[memory_idx] + cr) / 2
                    self.f_memory[memory_idx] = (self.f_memory[memory_idx] + f) / 2
                    if trial_value < personal_best_value[i]:
                        personal_best[i] = trial
                        personal_best_value[i] = trial_value
                        if trial_value < global_best_value:
                            global_best = trial
                            global_best_value = trial_value

            # Adaptive parameter tuning based on success rate
            self.success_rate = (successes / self.population_size) if successes > 0 else 0.1
            if self.success_rate < 0.2:
                self.w = min(0.9, self.w + 0.05)
            else:
                self.w = max(0.4, self.w - 0.05)

        return global_best