import numpy as np

class EnhancedMultiSwarmOptimizer:
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
        self.leader_selection_probability = 0.2
        self.local_search_probability = 0.2  # Increased probability for local search
        self.convergence_history = []

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
                        if np.random.rand() < 0.5:
                            selected_leader = self.swarm_best_positions[swarm]
                        else:
                            selected_leader = self.particles[swarm, np.random.randint(self.num_particles)]

                    inertia = self.inertia_weight * self.velocities[swarm, i]
                    cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    self.velocities[swarm, i] = inertia + cognitive + social

                    velocity_norm = np.linalg.norm(self.velocities[swarm, i])
                    if velocity_norm > 1.0:
                        self.velocities[swarm, i] /= velocity_norm

                    self.particles[swarm, i] += self.velocities[swarm, i]

                    for d in range(self.dim):
                        if self.particles[swarm, i, d] < lb[d] or self.particles[swarm, i, d] > ub[d]:
                            self.particles[swarm, i, d] = lb[d] + (ub[d] - self.particles[swarm, i, d]) % (ub[d] - lb[d])
                            if np.random.rand() < 0.5:
                                self.particles[swarm, i, d] = lb[d] + np.random.rand() * (ub[d] - lb[d]) 

                self.inertia_weight = max(0.4, self.inertia_weight * 0.99)
                self.c1 = max(1.0, self.c1 - 0.001)
                self.c2 = min(2.0, self.c2 + 0.001)

                if np.random.rand() < self.local_search_probability:
                    self._local_search(swarm)

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value

    def _is_converging(self):
        convergence_threshold = 1e-5
        return np.abs(self.global_best_value - np.min(self.swarm_best_values)) < convergence_threshold

    def _adjust_parameters(self):
        self.inertia_weight = np.random.uniform(0.8, 1.0)
        self.c1 = np.random.uniform(1.5, 2.0)
        self.c2 = np.random.uniform(1.5, 2.0)

    def _local_search(self, swarm):
        lb, ub = self.bounds
        best_particle_idx = np.argmin(self.swarm_best_values)
        candidate_pos = self.swarm_best_positions[best_particle_idx] + 0.1 * np.random.randn(self.dim)
        candidate_pos = np.clip(candidate_pos, lb, ub)
        candidate_value = func(candidate_pos)

        if candidate_value < self.swarm_best_values[swarm]:
            self.swarm_best_values[swarm] = candidate_value
            self.swarm_best_positions[swarm] = candidate_pos.copy()