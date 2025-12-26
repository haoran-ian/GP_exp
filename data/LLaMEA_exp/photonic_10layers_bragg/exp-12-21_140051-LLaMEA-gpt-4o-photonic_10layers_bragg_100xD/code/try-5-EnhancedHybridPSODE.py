import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.w_initial = 0.9
        self.w_final = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor_initial = 0.9
        self.mutation_factor_final = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_pos = np.copy(population)
        personal_best_val = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = personal_best_pos[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive inertia weight and mutation factor
            progress_ratio = evaluations / self.budget
            w = self.w_initial - (self.w_initial - self.w_final) * progress_ratio
            mutation_factor = self.mutation_factor_initial - (self.mutation_factor_initial - self.mutation_factor_final) * progress_ratio

            # PSO Component
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities 
                          + self.c1 * r1 * (personal_best_pos - population)
                          + self.c2 * r2 * (global_best_pos - population))
            population = np.clip(population + velocities, lb, ub)

            # DE Component with local search
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])
                trial_val = func(trial)
                evaluations += 1

                if trial_val < personal_best_val[i]:
                    personal_best_pos[i] = trial
                    personal_best_val[i] = trial_val

                    if trial_val < personal_best_val[global_best_idx]:
                        global_best_idx = i
                        global_best_pos = trial

                # Local search strategy: perturb the trial solution slightly
                if evaluations < self.budget:
                    local_trial = np.clip(trial + np.random.normal(0, 0.01, self.dim), lb, ub)
                    local_trial_val = func(local_trial)
                    evaluations += 1

                    if local_trial_val < trial_val:
                        personal_best_pos[i] = local_trial
                        personal_best_val[i] = local_trial_val

                        if local_trial_val < personal_best_val[global_best_idx]:
                            global_best_idx = i
                            global_best_pos = local_trial

                if evaluations >= self.budget:
                    break

        return global_best_pos