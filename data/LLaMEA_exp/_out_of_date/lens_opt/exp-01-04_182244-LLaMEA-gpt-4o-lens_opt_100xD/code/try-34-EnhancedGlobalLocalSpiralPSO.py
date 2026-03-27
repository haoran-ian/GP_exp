import numpy as np

class EnhancedGlobalLocalSpiralPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        num_particles = 10
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        angle_increment = 0.1
        initial_radius = 0.1
        radius = initial_radius
        max_velocity = 0.5

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                inertia = 0.9 * (1 - (self.evaluations / self.budget)**0.5)
                cognitive = 2.0 * r1 * (personal_best_positions[idx] - particles[idx])
                neighbor_best_position = self._get_neighbor_best(particles, idx, personal_best_values)
                social = 1.5 * r2 * (neighbor_best_position - particles[idx])
                velocities[idx] = inertia * velocities[idx] + cognitive + social
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

                if self.evaluations % 15 == 0:
                    random_direction = np.random.normal(0, 0.05, self.dim)
                    particles[idx] += random_direction
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if self.evaluations >= self.budget:
                    break
            
            if all(personal_best_values >= global_best_value):
                radius *= 1.2
                global_best_index = np.argmin(personal_best_values)
                global_best_position = personal_best_positions[global_best_index]
                global_best_value = personal_best_values[global_best_index]

        return global_best_position, global_best_value

    def _get_neighbor_best(self, particles, idx, personal_best_values):
        num_particles = len(particles)
        neighborhood_size = 3
        neighbors = [(idx + i) % num_particles for i in range(-neighborhood_size//2, neighborhood_size//2 + 1)]
        best_neighbor_idx = min(neighbors, key=lambda x: personal_best_values[x])
        return particles[best_neighbor_idx]