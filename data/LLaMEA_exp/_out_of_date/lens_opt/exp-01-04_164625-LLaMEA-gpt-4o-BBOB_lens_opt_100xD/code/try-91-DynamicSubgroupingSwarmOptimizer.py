import numpy as np

class DynamicSubgroupingSwarmOptimizer:
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
        self.dynamic_grouping_prob = 0.3  # Probability to regroup swarms
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.9

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

                    # Select leader with dynamic subgrouping
                    if np.random.rand() < self.dynamic_grouping_prob:
                        selected_leader = self.global_best_position
                    else:
                        selected_leader = self.swarm_best_positions[swarm]

                    # Update velocity and position
                    inertia = self.inertia_weight * self.velocities[swarm, i]
                    cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    self.velocities[swarm, i] = inertia + cognitive + social

                    # Adaptive velocity clamping
                    velocity_clamp = np.clip(np.abs(self.velocities[swarm, i]), 0, 0.5)
                    self.velocities[swarm, i] = np.sign(self.velocities[swarm, i]) * velocity_clamp

                    self.particles[swarm, i] += self.velocities[swarm, i]

                    # Ensure particles stay within bounds and use boundary reflection
                    for d in range(self.dim):
                        if self.particles[swarm, i, d] < lb[d] or self.particles[swarm, i, d] > ub[d]:
                            self.particles[swarm, i, d] = lb[d] + (ub[d] - self.particles[swarm, i, d]) % (ub[d] - lb[d])

                # Dynamic parameter adjustment
                if eval_count % (self.budget // 10) == 0:
                    self.inertia_weight = max(0.4, self.inertia_weight * 0.99)
                    self.c1, self.c2 = np.random.uniform(1.0, 2.0, 2)
                    self.dynamic_grouping_prob = 0.3 + 0.1 * np.sin(np.pi * eval_count / self.budget)  # Modified line

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value