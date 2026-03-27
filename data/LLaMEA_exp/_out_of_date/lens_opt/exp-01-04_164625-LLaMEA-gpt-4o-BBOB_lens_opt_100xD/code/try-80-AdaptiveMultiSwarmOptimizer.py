import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.num_particles = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.9
        self.bounds = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.swarm_best_positions = [None] * self.num_swarms
        self.swarm_best_values = [np.inf] * self.num_swarms
        self.particles = np.random.rand(self.num_swarms, self.num_particles, self.dim)
        self.velocities = np.zeros((self.num_swarms, self.num_particles, self.dim))
        self.leader_selection_probability = 0.25
        self.local_search_probability = 0.15
        self.dynamic_diversity_threshold = 0.1  # New parameter for diversity preservation
        self.shrink_factor = 0.95  # Used to reduce the search space adaptively

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        eval_count = 0
        lb, ub = self.bounds
        search_space_range = ub - lb
        self.particles = lb + search_space_range * self.particles

        while eval_count < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.num_particles):
                    particle_pos = self.particles[swarm, i]
                    particle_value = func(particle_pos)
                    eval_count += 1

                    if particle_value < self.swarm_best_values[swarm]:
                        self.swarm_best_values[swarm] = particle_value
                        self.swarm_best_positions[swarm] = particle_pos.copy()

                    if particle_value < self.global_best_value:
                        self.global_best_value = particle_value
                        self.global_best_position = particle_pos.copy()

                    if np.random.rand() < self.leader_selection_probability:
                        selected_leader = self.global_best_position
                    else:
                        if np.random.rand() < 0.5:
                            selected_leader = self.swarm_best_positions[swarm]
                        else:
                            selected_leader = self.particles[swarm, np.random.randint(self.num_particles)]

                    inertia = self.inertia_weight * self.velocities[swarm, i]
                    cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    self.velocities[swarm, i] = inertia + cognitive + social

                    # Normalize velocity
                    velocity_norm = np.linalg.norm(self.velocities[swarm, i])
                    if velocity_norm > 1.0:
                        self.velocities[swarm, i] /= velocity_norm

                    self.particles[swarm, i] += self.velocities[swarm, i]

                    for d in range(self.dim):
                        if self.particles[swarm, i, d] < lb[d] or self.particles[swarm, i, d] > ub[d]:
                            self.particles[swarm, i, d] = lb[d] + (ub[d] - self.particles[swarm, i, d]) % (ub[d] - lb[d])

                if np.std(self.particles[swarm], axis=0).mean() < self.dynamic_diversity_threshold:
                    self._preserve_diversity(swarm)

                self.inertia_weight = max(0.4, self.inertia_weight * 0.99)
                self.c1 = max(1.0, self.c1 - 0.001)
                self.c2 = min(2.0, self.c2 + 0.001)

                # Adaptive search space reduction
                if eval_count % (self.budget // 5) == 0:
                    lb = self.global_best_position - (self.global_best_position - lb) * (1 - self.shrink_factor)
                    ub = self.global_best_position + (ub - self.global_best_position) * (1 - self.shrink_factor)

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value

    def _preserve_diversity(self, swarm):
        # Introduce random particles to maintain diversity
        for i in range(self.num_particles):
            if np.random.rand() < self.local_search_probability:
                self.particles[swarm, i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)