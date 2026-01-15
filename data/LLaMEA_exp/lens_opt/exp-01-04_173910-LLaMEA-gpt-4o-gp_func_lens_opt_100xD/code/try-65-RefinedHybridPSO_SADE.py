import numpy as np

class RefinedHybridPSO_SADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.5  # cognitive component
        self.c2 = 2.0  # increased social component for better convergence
        self.w = 0.6  # reduced inertia weight to enhance exploration
        self.cr_initial = 0.9  # initial crossover rate in SADE
        self.f_initial = 0.5  # initial F factor in SADE
        self.cr_memory = [0.1, 0.3, 0.5, 0.9]
        self.f_memory = [0.4, 0.6, 0.8, 1.0]
        self.mutation_rate = 0.1  # new mutation rate control

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
            # Adaptive Mutation Control in PSO
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            vel = self.w * vel + self.c1 * r1 * (personal_best - pop) + self.c2 * r2 * (global_best - pop)
            
            # Introduce random walk for diversity
            mutation = np.random.uniform(-self.mutation_rate, self.mutation_rate, pop.shape)
            pop = pop + vel + mutation
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

            # Self-Adaptive Differential Evolution (SADE) with elite learning
            if evaluations < self.budget:
                for i in range(self.population_size):
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                    memory_idx = np.random.choice(len(self.cr_memory))
                    self.cr = self.cr_memory[memory_idx]
                    self.f = self.f_memory[memory_idx]
                    
                    # Mutation with elite learning
                    elite = global_best
                    mutant = np.clip(a + self.f * (b - c + elite - pop[i]), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, pop[i])
                    trial_value = func(trial)
                    evaluations += 1

                    # Selection
                    if trial_value < new_values[i]:
                        pop[i] = trial
                        new_values[i] = trial_value
                        self.cr_memory[memory_idx] = (self.cr_memory[memory_idx] + self.cr) / 2
                        self.f_memory[memory_idx] = (self.f_memory[memory_idx] + self.f) / 2
                        if trial_value < personal_best_value[i]:
                            personal_best[i] = trial
                            personal_best_value[i] = trial_value
                            if trial_value < global_best_value:
                                global_best = trial
                                global_best_value = trial_value

        return global_best