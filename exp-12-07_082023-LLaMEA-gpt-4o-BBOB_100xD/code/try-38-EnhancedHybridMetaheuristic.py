import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb, self.ub = -5.0, 5.0
        self.initial_population_size = 50
        self.min_population_size = 25
        self.population_size = self.initial_population_size
        self.velocity = np.zeros((self.population_size, dim))
        self.positions = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best_position = np.copy(self.positions)
        self.best_global_position = self.best_position[0]
        self.best_global_value = np.inf
        self.F = 0.6  # Initial DE scaling factor
        self.CR = 0.9  # Initial crossover probability for DE
        self.c1, self.c2 = 2.0, 2.0  # PSO cognitive and social coefficients
        self.w_max, self.w_min = 0.9, 0.4  # Max and min inertia weights
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

        for i in range(self.population_size):
            if values[i] < self.best_global_value:
                self.best_global_value = values[i]
                self.best_global_position = self.positions[i]

        chaotic_value = np.random.rand()
        while evaluations < self.budget:
            chaotic_value = self.chaotic_map(chaotic_value)
            self.population_size = max(self.min_population_size, self.population_size - 1)
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))  # Linear decay
            self.F = chaotic_value * (0.5 - 0.25 * np.tanh(evaluations / self.budget))
            adaptive_CR = self.CR * (1 - self.success_rate)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = inertia_weight * self.velocity[i] + \
                                  self.c1 * r1 * (self.best_position[i] - self.positions[i]) + \
                                  self.c2 * r2 * (self.best_global_position - self.positions[i])
                self.positions[i] += self.velocity[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.positions[indices]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3) + self.levy_flight(self.dim), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < adaptive_CR
                trial_vector = np.where(crossover, mutant_vector, self.positions[i])

                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < values[i]:
                    self.positions[i] = trial_vector
                    values[i] = trial_value
                    self.success_rate = 0.9 * self.success_rate + 0.1
                    if trial_value < self.best_global_value:
                        self.best_global_value = trial_value
                        self.best_global_position = trial_vector
                else:
                    self.success_rate = 0.9 * self.success_rate

            if np.std(values) < 1e-5:
                self.positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
                values = np.apply_along_axis(func, 1, self.positions)
                evaluations += self.population_size
                self.best_global_value = np.inf

        return self.best_global_value