import numpy as np

class EnhancedHybridPSO_SADE_AdaptiveGroups:
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
        self.num_subgroups = 3  # number of adaptive groups

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
            # Form subgroups to enhance local search exploitation
            group_size = self.population_size // self.num_subgroups
            subgroups = [pop[i:i + group_size] for i in range(0, self.population_size, group_size)]
            
            for sg in subgroups:
                sg_best_value = np.inf
                sg_best = None
                for ind in sg:
                    ind_value = func(ind)
                    if ind_value < sg_best_value:
                        sg_best_value = ind_value
                        sg_best = ind
                
                # Update velocities and positions (PSO)
                r1, r2 = np.random.rand(len(sg), self.dim), np.random.rand(len(sg), self.dim)
                sg_vel = self.w * vel[:len(sg)] + self.c1 * r1 * (personal_best[:len(sg)] - sg) + self.c2 * r2 * (sg_best - sg)
                sg_new = sg + sg_vel
                sg_new = np.clip(sg_new, lb, ub)

                # Evaluate new positions
                new_values = np.array([func(ind) for ind in sg_new])
                evaluations += len(sg)

                # Update personal and global bests
                improvement = new_values < personal_best_value[:len(sg)]
                personal_best[:len(sg)][improvement] = sg_new[improvement]
                personal_best_value[:len(sg)][improvement] = new_values[improvement]

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