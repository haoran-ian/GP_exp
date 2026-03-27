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
        self.leader_selection_probability = 0.25  # Adjusted leader selection probability
        self.local_search_probability = 0.15  # Adjusted local search probability

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

                    if np.random.rand() < self.leader_selection_probability:
                        selected_leader = self.global_best_position
                    else:
                        if np.random.rand() < 0.5:
                            selected_leader = self.swarm_best_positions[swarm]
                        else:
                            selected_leader = self.particles[swarm, np.random.randint(self.num_particles)]

                    # Adaptive parameter tuning
                    if eval_count % (self.budget // 10) == 0 and eval_count > 0:  # Dynamic adjustment
                        if not self._is_converging():  # if not converging, adjust parameters
                            self._adjust_parameters()

                    # Update velocity and position
                    inertia = self.inertia_weight * self.velocities[swarm, i]
                    cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    mutation = 0.01 * np.random.randn(self.dim)  # Added mutation
                    self.velocities[swarm, i] = inertia + cognitive + social + mutation

                    velocity_norm = np.linalg.norm(self.velocities[swarm, i])
                    if velocity_norm > 1.0:
                        self.velocities[swarm, i] /= velocity_norm

                    self.particles[swarm, i] += self.velocities[swarm, i]

                    # Ensure particles stay within bounds and use boundary reflection
                    for d in range(self.dim):
                        if self.particles[swarm, i, d] < lb[d] or self.particles[swarm, i, d] > ub[d]:
                            self.particles[swarm, i, d] = lb[d] + (ub[d] - self.particles[swarm, i, d]) % (ub[d] - lb[d])

                self.inertia_weight = max(0.4, self.inertia_weight * 0.99)
                self.c1 = max(1.0, self.c1 - 0.001)
                self.c2 = min(2.0, self.c2 + 0.001)

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value

    def _is_converging(self):
        # Check if the global best value is improving
        convergence_threshold = 1e-5
        return np.abs(self.global_best_value - np.min(self.swarm_best_values)) < convergence_threshold

    def _adjust_parameters(self):
        # Adjust parameters dynamically based on convergence status
        self.inertia_weight = np.random.uniform(0.8, 1.0)
        self.c1 = np.random.uniform(1.5, 2.0)
        self.c2 = np.random.uniform(1.5, 2.0)