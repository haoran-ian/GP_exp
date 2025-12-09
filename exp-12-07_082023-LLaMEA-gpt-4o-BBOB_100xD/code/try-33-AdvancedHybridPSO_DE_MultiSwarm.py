import numpy as np

class AdvancedHybridPSO_DE_MultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb, self.ub = -5.0, 5.0
        self.initial_pop_size = 40
        self.min_pop_size = 20
        self.pop_size = self.initial_pop_size
        self.vel = np.zeros((self.pop_size, dim))
        self.best_pos = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.global_best_pos = self.best_pos[0]
        self.global_best_val = np.inf
        self.F = 0.5
        self.CR = 0.9
        self.c1, self.c2 = 2.0, 2.0
        self.w_max, self.w_min = 0.9, 0.4
        self.success_rate = 0.0

    def logistic_map(self, x):
        return 3.9 * x * (1 - x)

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        evaluations = 0
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        values = np.apply_along_axis(func, 1, positions)
        evaluations += self.pop_size

        for i in range(self.pop_size):
            if values[i] < self.global_best_val:
                self.global_best_val = values[i]
                self.global_best_pos = positions[i]

        logistic_value = np.random.rand()
        swarm_diversity_threshold = 1e-5
        while evaluations < self.budget:
            logistic_value = self.logistic_map(logistic_value)
            self.pop_size = max(self.min_pop_size, self.pop_size - 1)
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            adaptive_CR = self.CR * (1 - self.success_rate)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.vel[i] = inertia_weight * self.vel[i] + \
                              (self.c1 * r1 * (self.best_pos[i] - positions[i]) + \
                               self.c2 * r2 * (self.global_best_pos - positions[i]))
                positions[i] += self.vel[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = positions[indices]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3) + self.levy_flight(self.dim), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < adaptive_CR
                trial_vector = np.where(crossover, mutant_vector, positions[i])

                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < values[i]:
                    positions[i] = trial_vector
                    values[i] = trial_value
                    self.success_rate = 0.9 * self.success_rate + 0.1 * 1
                    if trial_value < self.global_best_val:
                        self.global_best_val = trial_value
                        self.global_best_pos = trial_vector
                else:
                    self.success_rate = 0.9 * self.success_rate

                if np.std(values) < swarm_diversity_threshold:
                    positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
                    values = np.apply_along_axis(func, 1, positions)
                    evaluations += self.pop_size
                    self.global_best_val = np.inf

        return self.global_best_val