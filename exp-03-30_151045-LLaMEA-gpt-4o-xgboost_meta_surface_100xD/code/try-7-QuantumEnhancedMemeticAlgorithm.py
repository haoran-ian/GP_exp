import numpy as np

class QuantumEnhancedMemeticAlgorithm:
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

    def _quantum_mutation(self, position, bounds):
        mutation_prob = 1.0 / self.dim
        mutation_vector = np.random.uniform(bounds.lb, bounds.ub, self.dim)
        random_mask = np.random.rand(self.dim) < mutation_prob
        return np.where(random_mask, mutation_vector, position)

    def _refined_local_search(self, particle, func, bounds):
        step_size = 0.05 * (bounds.ub - bounds.lb)
        local_best = particle
        local_best_value = func(local_best)
        trials = 10
        for _ in range(trials):
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

                # Apply quantum mutation
                self.particles[i] = self._quantum_mutation(self.particles[i], bounds)

            # Apply refined local search on a fraction of the population
            for i in np.random.choice(self.population_size, self.population_size // 5, replace=False):
                local_best, local_best_value = self._refined_local_search(self.particles[i], func, bounds)
                if local_best_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = local_best_value
                    self.personal_best_positions[i] = local_best
                if local_best_value < self.global_best_value:
                    self.global_best_value = local_best_value
                    self.global_best_position = local_best

        return self.global_best_position, self.global_best_value