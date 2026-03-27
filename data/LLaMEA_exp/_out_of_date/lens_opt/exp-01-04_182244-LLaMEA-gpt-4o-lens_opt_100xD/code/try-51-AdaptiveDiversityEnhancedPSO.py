import numpy as np

class AdaptiveDiversityEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize parameters
        num_particles = 20  # Increase number of particles for better exploration
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        angle_increment = 0.15
        initial_radius = 0.1
        radius = initial_radius
        max_velocity = 0.5  # Adjust max velocity for finer control
        phase_shift = 0.7
        adaptive_mutation_freq = 8  # Increase mutation frequency for diversity

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                phase_factor = np.sin(2 * np.pi * phase_shift * self.evaluations / self.budget)
                inertia = 0.9 * np.exp(-0.005 * self.evaluations)  # Exponential decay for inertia
                
                # Use dynamic neighborhood to enhance exploration
                neighbor_indices = self._get_dynamic_neighbors(particles, idx, num_particles)
                neighbor_best_position = min(neighbor_indices, key=lambda i: personal_best_values[i])
                
                cognitive = 2.1 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.6 * r2 * (personal_best_positions[neighbor_best_position] - particles[idx])
                
                velocities[idx] = inertia * velocities[idx] + cognitive + 0.4 * social
                velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if idx == global_best_index:
                    angle = self.evaluations * angle_increment + np.random.normal(0, 0.05)
                    displacement = np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle)
                    ] + [0] * (self.dim - 2))
                    spiral_position = global_best_position + displacement[:self.dim]
                    spiral_position = np.clip(spiral_position, func.bounds.lb, func.bounds.ub)
                    particles[idx] = spiral_position

                value = func(particles[idx])
                self.evaluations += 1

                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]
                    radius = max(initial_radius, radius * 0.8)

                if self.evaluations % adaptive_mutation_freq == 0:
                    particles[idx] += np.random.normal(0, 0.03 + 0.1 * (self.evaluations / self.budget), self.dim)
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if self.evaluations >= self.budget:
                    break
            
            if all(personal_best_values >= global_best_value):
                radius *= 1.1

        return global_best_position, global_best_value

    def _get_dynamic_neighbors(self, particles, idx, num_particles):
        """Find a dynamic set of neighbor indices for better exploration."""
        neighborhood_size = 5  # Increase neighborhood size
        base_neighbors = [(idx + i) % num_particles for i in range(-neighborhood_size//2, neighborhood_size//2 + 1)]
        additional_indices = np.random.choice(num_particles, neighborhood_size, replace=False)
        return list(set(base_neighbors + list(additional_indices)))