import numpy as np

class AdaptiveMemoryCoopPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.randn(self.population_size, dim)
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0
        self.memory = np.full((self.population_size, dim), np.inf)
        self.memory_values = np.full(self.population_size, np.inf)

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def _update_memory(self, index, position, value):
        if value < self.memory_values[index]:
            self.memory[index] = position
            self.memory_values[index] = value

    def _cooperative_learning(self, index):
        partners = np.random.choice(np.delete(np.arange(self.population_size), index), 2, replace=False)
        cooperation_partner = min(partners, key=lambda x: self.memory_values[x])
        return self.memory[cooperation_partner]

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            lower_bound, upper_bound = bounds.lb, bounds.ub

            for i in range(self.population_size):
                value = func(self.particles[i])
                self.current_eval += 1

                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i]

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.particles[i]

                self._update_memory(i, self.particles[i], value)

            inertia_weight = self._adaptive_inertia_weight()

            for i in range(self.population_size):
                cooperative_partner_position = self._cooperative_learning(i)
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand() * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * np.random.rand() * (self.global_best_position - self.particles[i]) +
                                      0.5 * np.random.rand() * (cooperative_partner_position - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

        return self.global_best_position, self.global_best_value