import numpy as np

class RefinedMultiSwarmOptimizer:
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
        self.neighborhood_size = 5  # New parameter for neighborhood size

    def __call__(self, func):
        self.bounds = (func.bounds.lb, func.bounds.ub)
        eval_count = 0

        # Initialize particle positions within bounds
        lb, ub = self.bounds
        self.particles = lb + (ub - lb) * self.particles

        while eval_count < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.num_particles):
                    # Evaluate current particle
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

                    # Dynamic leader selection
                    selected_leader = self.global_best_position if np.random.rand() < self.leader_selection_probability else self.swarm_best_positions[swarm]

                    # Neighborhood-based velocity adjustment
                    start_idx = max(0, i - self.neighborhood_size // 2)
                    end_idx = min(self.num_particles, i + self.neighborhood_size // 2)
                    neighborhood_best = np.min([
                        func(self.particles[swarm, j]) for j in range(start_idx, end_idx)
                    ])

                    # Update velocity and position
                    inertia = self.inertia_weight * self.velocities[swarm, i]
                    cognitive = self.c1 * np.random.rand(self.dim) * (self.swarm_best_positions[swarm] - particle_pos)
                    social = self.c2 * np.random.rand(self.dim) * (selected_leader - particle_pos)
                    neighborhood_influence = np.random.rand(self.dim) * (neighborhood_best - particle_pos)  # New influence

                    self.velocities[swarm, i] = inertia + cognitive + social + neighborhood_influence

                    # Apply variable velocity scaling
                    velocity_norm = np.linalg.norm(self.velocities[swarm, i])
                    if velocity_norm > 1.0:
                        self.velocities[swarm, i] /= velocity_norm

                    self.particles[swarm, i] += self.velocities[swarm, i]

                    # Ensure particles stay within bounds
                    np.clip(self.particles[swarm, i], lb, ub, out=self.particles[swarm, i])

                # Update adaptive parameters
                self.inertia_weight = max(0.4, self.inertia_weight * 0.99)
                self.c1 = max(1.0, self.c1 - 0.001)
                self.c2 = min(2.0, self.c2 + 0.001)

            if eval_count >= self.budget:
                break

        return self.global_best_position, self.global_best_value