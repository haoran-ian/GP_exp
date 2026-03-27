import numpy as np

class RefinedHybridQPSO_SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.7  # inertia weight
        self.cr = 0.9  # initial crossover rate in SADE
        self.f = 0.5  # initial F factor in SADE
        self.cr_memory = [0.1, 0.2, 0.5, 0.9]
        self.f_memory = [0.4, 0.6, 0.8, 1.0]
        self.mb = np.zeros(dim)  # Mean best position

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_value = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            # Quantum-behaved PSO update
            self.mb = np.mean(personal_best, axis=0)
            for i in range(self.population_size):
                u = np.random.rand(self.dim)
                p = self.c1 * personal_best[i] + self.c2 * global_best
                beta = np.random.uniform(0, 1, self.dim)
                pop[i] = self.mb + beta * np.abs(p - self.mb) * np.log(1/u)

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
                        if trial_value < personal_best_value[i]:
                            personal_best[i] = trial
                            personal_best_value[i] = trial_value
                            if trial_value < global_best_value:
                                global_best = trial
                                global_best_value = trial_value

        return global_best