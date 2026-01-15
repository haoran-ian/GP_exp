import numpy as np

class EnhancedDynamicMultiphasePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize parameters
        num_particles = 12
        num_swarms = 2
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_swarms, num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_swarms, num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([[func(p) for p in swarm] for swarm in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions.flatten()[global_best_index]
        global_best_value = np.min(personal_best_values)
        angle_increment = 0.1
        initial_radius = 0.1
        radius = initial_radius
        max_velocity = 0.5
        adaptive_learning_rate = 0.1  # Learning rate starts adaptive changes

        while self.evaluations < self.budget:
            for swarm_idx in range(num_swarms):
                for idx in range(num_particles):
                    # Update velocity and position using PSO formula with adaptive learning rate
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    
                    # Adaptive inertia decay with learning rate adjustment
                    inertia = 0.9 * (1 - (self.evaluations / self.budget)**0.5)
                    cognitive = 2.0 * r1 * (personal_best_positions[swarm_idx, idx] - particles[swarm_idx, idx])
                    neighbor_best_position = self._get_neighbor_best(particles[swarm_idx], idx, personal_best_values[swarm_idx])
                    social = 1.5 * r2 * (neighbor_best_position - particles[swarm_idx, idx])
                    
                    # Adaptive velocity scaling
                    velocities[swarm_idx, idx] = inertia * velocities[swarm_idx, idx] + cognitive + 0.5 * social
                    velocities[swarm_idx, idx] = np.clip(velocities[swarm_idx, idx], -max_velocity, max_velocity)
                    particles[swarm_idx, idx] += velocities[swarm_idx, idx]
                    particles[swarm_idx, idx] = np.clip(particles[swarm_idx, idx], func.bounds.lb, func.bounds.ub)

                    # Enhanced stochastic spiral exploration
                    if idx == global_best_index:
                        angle = self.evaluations * angle_increment + np.random.normal(0, 0.05)
                        displacement = np.array([
                            radius * np.cos(angle),
                            radius * np.sin(angle)
                        ] + [0] * (self.dim - 2))
                        spiral_position = global_best_position + displacement[:self.dim]
                        spiral_position = np.clip(spiral_position, func.bounds.lb, func.bounds.ub)
                        particles[swarm_idx, idx] = spiral_position

                    # Evaluate particles
                    value = func(particles[swarm_idx, idx])
                    self.evaluations += 1

                    # Update personal and global bests
                    if value < personal_best_values[swarm_idx, idx]:
                        personal_best_values[swarm_idx, idx] = value
                        personal_best_positions[swarm_idx, idx] = particles[swarm_idx, idx]
                    if value < global_best_value:
                        global_best_value = value
                        global_best_position = particles[swarm_idx, idx]
                        radius = max(initial_radius, radius * 0.8)

                    # Adaptive learning rate adjustment based on progress
                    if np.random.rand() < adaptive_learning_rate:
                        particles[swarm_idx, idx] += np.random.normal(0, 0.1 * (1 - self.evaluations / self.budget), self.dim)
                        particles[swarm_idx, idx] = np.clip(particles[swarm_idx, idx], func.bounds.lb, func.bounds.ub)

                    if self.evaluations >= self.budget:
                        break
                
                # Increase spiral search radius if no improvement in this swarm
                if all(personal_best_values[swarm_idx] >= global_best_value):
                    radius *= 1.1  # Slightly increase less aggressively

        return global_best_position, global_best_value

    def _get_neighbor_best(self, particles, idx, personal_best_values):
        """Find the best personal best position among the neighbors."""
        num_particles = len(particles)
        neighborhood_size = 3
        neighbors = [(idx + i) % num_particles for i in range(-neighborhood_size//2, neighborhood_size//2 + 1)]
        best_neighbor_idx = min(neighbors, key=lambda x: personal_best_values[x])
        return particles[best_neighbor_idx]