import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb, self.ub = -5.0, 5.0
        self.population_size = 40  # Typically small for PSO-DE hybrid
        self.velocity = np.zeros((self.population_size, dim))
        self.best_position = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best_global_position = self.best_position[0]
        self.best_global_value = np.inf
        self.F = 0.5  # Differential Evolution scaling factor
        self.CR = 0.9  # Crossover probability for DE
        self.c1, self.c2 = 2.0, 2.0  # PSO cognitive and social coefficients
        self.w_max, self.w_min = 0.9, 0.4  # Max and min inertia weights

    def __call__(self, func):
        evaluations = 0
        positions = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        values = np.apply_along_axis(func, 1, positions)
        evaluations += self.population_size

        for i in range(self.population_size):
            if values[i] < self.best_global_value:
                self.best_global_value = values[i]
                self.best_global_position = positions[i]

        while evaluations < self.budget:
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))  # Adaptive inertia weight
            for i in range(self.population_size):
                # Particle Swarm Optimization update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = inertia_weight * self.velocity[i] + \
                                  self.c1 * r1 * (self.best_position[i] - positions[i]) + \
                                  self.c2 * r2 * (self.best_global_position - positions[i])
                positions[i] += self.velocity[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                # Differential Evolution mutation and crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = positions[indices]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant_vector, positions[i])

                # Evaluation and selection
                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < values[i]:
                    positions[i] = trial_vector
                    values[i] = trial_value
                    if trial_value < self.best_global_value:
                        self.best_global_value = trial_value
                        self.best_global_position = trial_vector

                if evaluations >= self.budget:
                    break

        return self.best_global_value