import numpy as np

class AdvancedDynamicSubgroupingSwarmOptimizer:
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
        self.c1_range = (1.0, 2.0)
        self.c2_range = (1.0, 2.0)
        self.inertia_weight_range = (0.4, 0.9)

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        eval_count = 0

        lb, ub = self.bounds
        self.particles = lb + (ub - lb) * self.particles

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

                    if np.random.rand() < self.dynamic_grouping_prob:
                        selected_leader = self.global_best_position
                    else:
                        selected_leader = self.swarm_best_positions[swarm]

                    inertia_weight = np.random.uniform(*self.inertia_weight_range)
                    c1 = np.random.uniform(*self.c1_range)
                    c2 = np.random.uniform(*self.c2_range)
                    
                    inertia = inertia_weight * self.velocities[swarm, i]
                    cognitive = c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    self.velocities[swarm, i] = inertia + cognitive + social

                    self.velocities[swarm, i] = np.clip(self.velocities[swarm, i], -0.5, 0.5)
                    self.particles[swarm, i] += self.velocities[swarm, i]

                    for d in range(self.dim):
                        if self.particles[swarm, i, d] < lb[d] or self.particles[swarm, i, d] > ub[d]:
                            self.particles[swarm, i, d] = lb[d] + (ub[d] - self.particles[swarm, i, d]) % (ub[d] - lb[d])

                if eval_count >= self.budget:
                    break

        return self.global_best_position, self.global_best_value