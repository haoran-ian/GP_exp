import numpy as np

class EnhancedMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.num_swarms = 5
        self.swarm_size = self.population_size // self.num_swarms
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.randn(self.population_size, dim)
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_positions = np.zeros((self.num_swarms, dim))
        self.global_best_values = np.full(self.num_swarms, np.inf)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))
        
    def _adaptive_local_search(self, particle, func, bounds):
        step_size = 0.1 * (bounds.ub - bounds.lb)
        local_best = particle
        local_best_value = func(local_best)
        trials = 5 + int(5 * (1 - (self.current_eval / self.budget)))
        for _ in range(trials):
            candidate = local_best + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_value = func(candidate)
            if candidate_value < local_best_value:
                local_best = candidate
                local_best_value = candidate_value
        return local_best, local_best_value

    def _adaptive_mutation(self, particle, bounds):
        mutation_rate = 0.1 * (1 - (self.current_eval / self.budget))
        mutation_vector = np.random.normal(0, mutation_rate, self.dim)
        mutated_particle = particle + mutation_vector
        return np.clip(mutated_particle, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            for swarm_id in range(self.num_swarms):
                swarm_start = swarm_id * self.swarm_size
                swarm_end = swarm_start + self.swarm_size

                for i in range(swarm_start, swarm_end):
                    # Evaluate current particle
                    value = func(self.particles[i])
                    self.current_eval += 1

                    # Update personal best
                    if value < self.personal_best_values[i]:
                        self.personal_best_values[i] = value
                        self.personal_best_positions[i] = self.particles[i]

                    # Update swarm best
                    if value < self.global_best_values[swarm_id]:
                        self.global_best_values[swarm_id] = value
                        self.global_best_positions[swarm_id] = self.particles[i]

                inertia_weight = self._adaptive_inertia_weight()

                for i in range(swarm_start, swarm_end):
                    # Update velocity and position
                    r1, r2 = np.random.rand(2)
                    self.velocities[i] = (inertia_weight * self.velocities[i] +
                                          self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                          self.c2 * r2 * (self.global_best_positions[swarm_id] - self.particles[i]) * np.random.uniform(0.9, 1.1))
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

                # Apply adaptive local search on a fraction of the swarm
                for i in np.random.choice(range(swarm_start, swarm_end), self.swarm_size // 10, replace=False):
                    local_best, local_best_value = self._adaptive_local_search(self.particles[i], func, bounds)
                    if local_best_value < self.personal_best_values[i]:
                        self.personal_best_values[i] = local_best_value
                        self.personal_best_positions[i] = local_best
                    if local_best_value < self.global_best_values[swarm_id]:
                        self.global_best_values[swarm_id] = local_best_value
                        self.global_best_positions[swarm_id] = local_best

                # Apply adaptive mutation
                for i in range(swarm_start, swarm_end):
                    mutated_particle = self._adaptive_mutation(self.particles[i], bounds)
                    mutated_value = func(mutated_particle)
                    self.current_eval += 1
                    if mutated_value < self.personal_best_values[i]:
                        self.personal_best_values[i] = mutated_value
                        self.personal_best_positions[i] = mutated_particle
                    if mutated_value < self.global_best_values[swarm_id]:
                        self.global_best_values[swarm_id] = mutated_value
                        self.global_best_positions[swarm_id] = mutated_particle

        # Return the best global position and value among all swarms
        best_swarm_idx = np.argmin(self.global_best_values)
        return self.global_best_positions[best_swarm_idx], self.global_best_values[best_swarm_idx]