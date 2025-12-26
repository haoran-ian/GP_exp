import numpy as np

class RefinedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.w_max = 0.9  # max inertia weight
        self.w_min = 0.2  # min inertia weight (reduced from 0.4 for more exploration)
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
            w = self.w_max - (self.w_max - self.w_min) * (evaluations**0.5 / self.budget**0.5)
            adaptive_lr = 0.1 + 0.9 * (1 - evaluations / self.budget)
            self.mutation_factor = 0.5 + 0.5 * np.sin(evaluations * np.pi / self.budget) * adaptive_lr
            
            # Updated line for damping cognitive and social components
            c_damp = 1 - evaluations / self.budget
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities 
                          + self.c1 * c_damp * r1 * (personal_best_pos - population)
                          + self.c2 * c_damp * r2 * (global_best_pos - population))
            velocities *= (0.1 + 0.9 * (1 - evaluations / self.budget)**2)
            population = np.clip(population + velocities, lb, ub)

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                chaos_factor = np.sin(evaluations * np.pi / self.budget)
                mutant = np.clip(a + self.mutation_factor * chaos_factor * (b - c), lb, ub)
                crossover_rate_varied = self.crossover_rate * (1 - evaluations / self.budget)
                crossover = np.random.rand(self.dim) < crossover_rate_varied
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