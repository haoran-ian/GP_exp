import numpy as np

class RefinedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.w_max = 0.9  # max inertia weight
        self.w_min = 0.4  # min inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def chaotic_map_init(self, N, D):
        chaos_seq = np.zeros((N, D))
        x0 = np.random.rand()
        for d in range(D):
            x = x0
            for n in range(N):
                x = 4 * x * (1 - x)
                chaos_seq[n, d] = x
        return chaos_seq

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        chaos_population = self.chaotic_map_init(self.population_size, self.dim)
        population = lb + (ub - lb) * chaos_population
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_pos = np.copy(population)
        personal_best_val = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = personal_best_pos[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * ((evaluations / self.budget)**2)  # Nonlinear inertia weight
            self.mutation_factor = 0.5 + 0.3 * np.sin(evaluations * np.pi / self.budget)  # Dynamic mutation factor
            # PSO Component
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities 
                          + self.c1 * r1 * (personal_best_pos - population)
                          + self.c2 * r2 * (global_best_pos - population))
            population = np.clip(population + velocities, lb, ub)

            # DE Component
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
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