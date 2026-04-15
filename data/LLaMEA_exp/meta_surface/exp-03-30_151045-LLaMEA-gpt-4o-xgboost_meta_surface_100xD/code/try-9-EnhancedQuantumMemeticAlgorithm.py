import numpy as np

class EnhancedQuantumMemeticAlgorithm:
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
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / (self.budget)))

    def _levy_flight_mutation(self, position, bounds):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / beta)
        mutation_step = 0.01 * step * (bounds.ub - bounds.lb)
        mutated_position = position + mutation_step
        return np.clip(mutated_position, bounds.lb, bounds.ub)

    def _dynamic_local_search(self, particle, func, bounds):
        step_size = 0.1 * (bounds.ub - bounds.lb)
        local_best = particle
        local_best_value = func(local_best)
        neighborhood_size = max(1, int(self.dim * 0.1))
        
        for _ in range(neighborhood_size):
            candidate = local_best + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_value = func(candidate)
            if candidate_value < local_best_value:
                local_best = candidate
                local_best_value = candidate_value
        return local_best, local_best_value

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            for i in range(self.population_size):
                # Evaluate current particle
                value = func(self.particles[i])
                self.current_eval += 1

                # Update personal best
                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i]

                # Update global best
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.particles[i]

            inertia_weight = self._adaptive_inertia_weight()

            for i in range(self.population_size):
                # Update velocity and position
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

                # Apply Lévy flight mutation
                self.particles[i] = self._levy_flight_mutation(self.particles[i], bounds)

            # Apply dynamic local search on a fraction of the population
            for i in np.random.choice(self.population_size, self.population_size // 5, replace=False):
                local_best, local_best_value = self._dynamic_local_search(self.particles[i], func, bounds)
                if local_best_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = local_best_value
                    self.personal_best_positions[i] = local_best
                if local_best_value < self.global_best_value:
                    self.global_best_value = local_best_value
                    self.global_best_position = local_best

        return self.global_best_position, self.global_best_value