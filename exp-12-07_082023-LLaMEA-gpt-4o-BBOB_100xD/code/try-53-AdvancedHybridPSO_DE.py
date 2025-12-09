import numpy as np

class AdvancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb, self.ub = -5.0, 5.0
        self.initial_population_size = 40
        self.min_population_size = 20
        self.population_size = self.initial_population_size
        self.velocity = np.zeros((self.population_size, dim))
        self.positions = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best_positions = self.positions.copy()
        self.global_leaders = self.positions[np.argsort([np.inf] * self.population_size)[:3]]
        self.global_best_value = np.inf
        self.F = 0.5
        self.CR = 0.9
        self.c1, self.c2 = 2.0, 2.0
        self.w_max, self.w_min = 0.9, 0.4
        self.mutation_rate = 0.1
        self.success_rate = 0.0

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        evaluations = 0
        values = np.apply_along_axis(func, 1, self.positions)
        evaluations += self.population_size

        if evaluations >= self.budget:
            return self.global_best_value

        self.update_global_leaders(values)

        while evaluations < self.budget:
            chaotic_value = np.random.rand()
            chaotic_value = self.chaotic_map(chaotic_value)
            self.population_size = max(self.min_population_size, self.population_size - 1)
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * np.sin(np.pi * (evaluations / self.budget)))

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                leader = self.global_leaders[np.random.randint(3)]
                self.velocity[i] = inertia_weight * self.velocity[i] + \
                                  self.c1 * r1 * (self.best_positions[i] - self.positions[i]) + \
                                  self.c2 * r2 * (leader - self.positions[i])
                self.positions[i] += self.velocity[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.positions[indices]
                gaussian_noise_scale = 0.1 * (1 - evaluations / self.budget) * (1 + self.success_rate)
                mutant_vector = np.clip(x1 + self.F * (x2 - x3) + self.levy_flight(self.dim) + np.random.normal(0, gaussian_noise_scale, self.dim), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant_vector, self.positions[i])

                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < values[i]:
                    self.positions[i] = trial_vector
                    values[i] = trial_value
                    self.success_rate = 0.9 * self.success_rate + 0.1 * 1
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.update_global_leaders(values)
                else:
                    self.success_rate = 0.9 * self.success_rate

                if evaluations < self.budget:
                    local_search_vector = self.positions[i] + self.mutation_rate * (self.global_leaders[0] - self.positions[i])
                    local_search_vector = np.clip(local_search_vector, self.lb, self.ub)
                    local_value = func(local_search_vector)
                    evaluations += 1
                    if local_value < values[i]:
                        self.positions[i] = local_search_vector
                        values[i] = local_value
                        if local_value < self.global_best_value:
                            self.global_best_value = local_value
                            self.update_global_leaders(values)

                if evaluations >= self.budget:
                    break

            if np.std(values) < 1e-5:
                self.positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
                values = np.apply_along_axis(func, 1, self.positions)
                evaluations += self.population_size
                self.global_best_value = np.inf
                self.update_global_leaders(values)

        return self.global_best_value

    def update_global_leaders(self, values):
        sorted_indices = np.argsort(values)
        self.global_leaders = self.positions[sorted_indices[:3]]
        self.global_best_value = values[sorted_indices[0]]