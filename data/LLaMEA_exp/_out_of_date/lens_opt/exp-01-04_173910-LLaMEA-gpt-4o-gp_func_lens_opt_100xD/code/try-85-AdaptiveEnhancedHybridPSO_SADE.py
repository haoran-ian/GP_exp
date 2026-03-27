import numpy as np

class AdaptiveEnhancedHybridPSO_SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.9  # starting inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.cr = 0.9  # initial crossover rate in SADE
        self.f = 0.5  # initial F factor in SADE
        self.cr_memory = [0.1, 0.2, 0.5, 0.9]
        self.f_memory = [0.4, 0.6, 0.8, 1.0]
        self.dynamic_population = True

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.initial_population_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_value = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.initial_population_size
        current_population_size = self.initial_population_size

        while evaluations < self.budget:
            # Adjust inertia weight adaptively based on evaluations
            self.w = self.w_min + (0.9 - self.w_min) * ((self.budget - evaluations) / self.budget)

            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(current_population_size, self.dim), np.random.rand(current_population_size, self.dim)
            vel = self.w * vel + self.c1 * r1 * (personal_best[:current_population_size] - pop[:current_population_size]) + self.c2 * r2 * (global_best - pop[:current_population_size])
            pop[:current_population_size] = pop[:current_population_size] + vel
            pop[:current_population_size] = np.clip(pop[:current_population_size], lb, ub)

            # Evaluate new positions
            new_values = np.array([func(ind) for ind in pop[:current_population_size]])
            evaluations += current_population_size

            # Update personal and global bests
            improvement = new_values < personal_best_value[:current_population_size]
            personal_best[:current_population_size][improvement] = pop[:current_population_size][improvement]
            personal_best_value[:current_population_size][improvement] = new_values[improvement]

            if np.min(personal_best_value[:current_population_size]) < global_best_value:
                global_best = personal_best[np.argmin(personal_best_value[:current_population_size])]
                global_best_value = np.min(personal_best_value[:current_population_size])

            # Self-Adaptive Differential Evolution (SADE)
            if evaluations < self.budget:
                for i in range(current_population_size):
                    indices = list(range(current_population_size))
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
            
            # Dynamically adjust population size
            if self.dynamic_population and evaluations < self.budget:
                if evaluations / self.budget < 0.5:
                    current_population_size = min(self.initial_population_size, len(pop))
                else:
                    current_population_size = max(4, int(self.initial_population_size * (1 - (evaluations / self.budget))))

        return global_best