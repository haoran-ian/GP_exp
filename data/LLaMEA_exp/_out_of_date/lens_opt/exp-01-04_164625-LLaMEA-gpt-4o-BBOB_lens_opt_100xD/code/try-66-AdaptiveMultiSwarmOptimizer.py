import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.num_particles = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.9
        self.bounds = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.swarm_best_positions = [None] * self.num_swarms
        self.swarm_best_values = [np.inf] * self.num_swarms
        self.particles = np.random.rand(self.num_swarms, self.num_particles, self.dim)
        self.velocities = np.zeros((self.num_swarms, self.num_particles, self.dim))
        self.leader_selection_probability = 0.3
        self.local_search_probability = 0.2
        self.momentum_factor = 0.5

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

                    if np.random.rand() < self.leader_selection_probability:
                        selected_leader = self.global_best_position
                    else:
                        local_leader = np.random.choice([self.swarm_best_positions[swarm], self.global_best_position])
                        selected_leader = local_leader

                    inertia = self.inertia_weight * self.velocities[swarm, i]
                    cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    self.velocities[swarm, i] = inertia + cognitive + social

                    self.particles[swarm, i] += self.velocities[swarm, i] * self.momentum_factor

                    for d in range(self.dim):
                        if self.particles[swarm, i, d] < lb[d] or self.particles[swarm, i, d] > ub[d]:
                            self.particles[swarm, i, d] = np.clip(self.particles[swarm, i, d], lb[d], ub[d])

                self.inertia_weight = max(0.4, self.inertia_weight * 0.99)
                if eval_count % (self.budget // 10) == 0:
                    self._adjust_parameters()

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value

    def _adjust_parameters(self):
        self.inertia_weight = np.random.uniform(0.7, 0.9)
        self.c1 = np.random.uniform(1.5, 2.5)
        self.c2 = np.random.uniform(1.5, 2.5)
        self.momentum_factor = np.random.uniform(0.4, 0.6)