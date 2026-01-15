import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize parameters
        num_particles = 20
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        max_velocity = 0.5
        adaptive_mutation_freq = 15

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = 0.5 + 0.4 * np.cos((self.evaluations / self.budget) * np.pi)

                # Dynamic neighborhood topology
                neighbor_indices = self._get_dynamic_neighbors(particles, idx, num_particles)
                neighbor_best_position = min(neighbor_indices, key=lambda i: personal_best_values[i])
                
                cognitive = 2.0 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.5 * r2 * (personal_best_positions[neighbor_best_position] - particles[idx])
                
                velocities[idx] = inertia * velocities[idx] + cognitive + social
                velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if idx == global_best_index:
                    particles[idx] += np.random.normal(0, 0.02, self.dim)
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                value = func(particles[idx])
                self.evaluations += 1

                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]

                if self.evaluations % adaptive_mutation_freq == 0:
                    velocities[idx] += np.random.normal(0, 0.05, self.dim)
                    velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)

                if self.evaluations >= self.budget:
                    break
            
            if all(personal_best_values >= global_best_value):
                velocities *= 0.9

        return global_best_position, global_best_value

    def _get_dynamic_neighbors(self, particles, idx, num_particles):
        """Find a dynamic set of neighbor indices for better exploration."""
        neighborhood_size = 5
        neighbor_indices = np.random.choice(num_particles, neighborhood_size, replace=False)
        return neighbor_indices