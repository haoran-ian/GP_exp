import numpy as np

class MultiPhaseAdaptiveVelocityPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize parameters
        num_particles = 15
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        max_velocity = 0.6

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cycle_factor = np.sin(2 * np.pi * (self.evaluations / self.budget))
                inertia = 0.5 + 0.4 * cycle_factor  # Adaptive inertia

                # Use adaptive neighborhood diversity to enhance exploration
                neighbor_indices = self._get_diverse_neighbors(particles, idx, num_particles)
                neighbor_best_position = min(neighbor_indices, key=lambda i: personal_best_values[i])
                
                cognitive = 2.1 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.4 * r2 * (personal_best_positions[neighbor_best_position] - particles[idx])
                
                # Introduce velocity oscillation
                velocity_oscillation = max_velocity * cycle_factor
                velocities[idx] = inertia * velocities[idx] + cognitive + social + velocity_oscillation
                velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                value = func(particles[idx])
                self.evaluations += 1

                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]

                if self.evaluations >= self.budget:
                    break
            
        return global_best_position, global_best_value

    def _get_diverse_neighbors(self, particles, idx, num_particles):
        """Find a diverse set of neighbor indices for better exploration."""
        neighborhood_size = 4
        base_neighbors = [(idx + i) % num_particles for i in range(-neighborhood_size//2, neighborhood_size//2 + 1)]
        additional_indices = np.random.choice(num_particles, neighborhood_size, replace=False)
        return list(set(base_neighbors + list(additional_indices)))