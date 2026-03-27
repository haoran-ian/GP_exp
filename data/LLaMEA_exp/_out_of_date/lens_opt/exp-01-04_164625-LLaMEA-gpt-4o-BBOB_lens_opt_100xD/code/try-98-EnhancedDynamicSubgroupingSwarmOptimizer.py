import numpy as np

class EnhancedDynamicSubgroupingSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.num_particles = 20
        self.bounds = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.swarm_best_positions = [None] * self.num_swarms
        self.swarm_best_values = [np.inf] * self.num_swarms
        self.particles = np.random.rand(self.num_swarms, self.num_particles, self.dim)
        self.velocities = np.zeros((self.num_swarms, self.num_particles, self.dim))
        self.dynamic_grouping_prob = 0.3
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.9
        self.learning_rate_decay = 0.99  # New adaptive decay factor

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        eval_count = 0

        # Initialize particle positions within bounds
        lb, ub = self.bounds
        self.particles = lb + (ub - lb) * self.particles

        while eval_count < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.num_particles):
                    particle_pos = self.particles[swarm, i]
                    particle_value = func(particle_pos)
                    eval_count += 1

                    # Update swarm bests
                    if particle_value < self.swarm_best_values[swarm]:
                        self.swarm_best_values[swarm] = particle_value
                        self.swarm_best_positions[swarm] = particle_pos.copy()

                    # Update global best
                    if particle_value < self.global_best_value:
                        self.global_best_value = particle_value
                        self.global_best_position = particle_pos.copy()

                    # Select leader with dynamic subgrouping and adaptive selecting
                    if np.random.rand() < self.dynamic_grouping_prob:
                        selected_leader = self.global_best_position
                    else:
                        selected_leader = self.swarm_best_positions[swarm]

                    # Update velocity and position with adaptive learning
                    inertia = self.inertia_weight * self.velocities[swarm, i]
                    cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    self.velocities[swarm, i] = inertia + cognitive + social

                    # Adaptive velocity clamping with self-adaptive strategy
                    velocity_norm = np.linalg.norm(self.velocities[swarm, i])
                    if velocity_norm > 0.5:
                        self.velocities[swarm, i] *= (0.5 / velocity_norm)

                    self.particles[swarm, i] += self.velocities[swarm, i]

                    # Ensure particles stay within bounds and use boundary reflection
                    for d in range(self.dim):
                        if self.particles[swarm, i, d] < lb[d]:
                            self.particles[swarm, i, d] = lb[d] + (lb[d] - self.particles[swarm, i, d])
                        elif self.particles[swarm, i, d] > ub[d]:
                            self.particles[swarm, i, d] = ub[d] - (self.particles[swarm, i, d] - ub[d])

                # Dynamic parameter adjustment with adaptive learning rate decay
                if eval_count % (self.budget // 10) == 0:
                    self.inertia_weight = max(0.4, self.inertia_weight * self.learning_rate_decay)
                    self.c1, self.c2 = np.random.uniform(1.0, 2.0, 2) * self.learning_rate_decay

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value