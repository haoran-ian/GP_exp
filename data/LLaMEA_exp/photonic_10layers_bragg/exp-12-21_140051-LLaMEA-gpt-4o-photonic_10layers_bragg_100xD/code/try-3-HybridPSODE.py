import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.mutation_factor = 0.9  # modified mutation factor
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
            # PSO Component
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities 
                          + self.c1 * r1 * (personal_best_pos - population)
                          + self.c2 * r2 * (global_best_pos - population))
            population = np.clip(population + velocities, lb, ub)

            # DE Component
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(population[a] + self.mutation_factor * (population[b] - population[c]), lb, ub)  # enhanced mutation
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

                if evaluations >= self.budget:
                    break

        return global_best_pos