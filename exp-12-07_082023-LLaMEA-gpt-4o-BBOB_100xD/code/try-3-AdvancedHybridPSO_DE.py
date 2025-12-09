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
        self.best_position = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best_global_position = self.best_position[0]
        self.best_global_value = np.inf
        self.F = 0.5  
        self.CR = 0.9  
        self.c1, self.c2 = 2.0, 2.0  
        self.w_max, self.w_min = 0.9, 0.4  

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
            self.population_size = max(self.min_population_size, self.population_size - 1)
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
            adaptive_F = self.F * (1 - (evaluations / self.budget))
            adaptive_CR = self.CR * (1 - (evaluations / self.budget))

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = inertia_weight * self.velocity[i] + \
                                  self.c1 * r1 * (self.best_position[i] - positions[i]) + \
                                  self.c2 * r2 * (self.best_global_position - positions[i])
                positions[i] += self.velocity[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = positions[indices]
                mutant_vector = np.clip(x1 + adaptive_F * (x2 - x3), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < adaptive_CR
                trial_vector = np.where(crossover, mutant_vector, positions[i])

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

            # Self-adaptive learning mechanism
            if evaluations % (self.budget // 10) == 0: 
                successful_trials = values < self.best_global_value
                if np.sum(successful_trials) > 0:
                    F_success = np.mean(self.F[successful_trials])
                    CR_success = np.mean(self.CR[successful_trials])
                    self.F = 0.8 * self.F + 0.2 * F_success
                    self.CR = 0.8 * self.CR + 0.2 * CR_success

            # Diversity preservation strategy
            if evaluations % (self.budget // 5) == 0:
                diversity = np.std(positions, axis=0)
                if np.all(diversity < 0.1):
                    new_positions = np.random.uniform(self.lb, self.ub, (self.population_size // 2, self.dim))
                    positions[-(self.population_size // 2):] = new_positions
                    values[-(self.population_size // 2):] = np.apply_along_axis(func, 1, new_positions)
                    evaluations += self.population_size // 2

        return self.best_global_value