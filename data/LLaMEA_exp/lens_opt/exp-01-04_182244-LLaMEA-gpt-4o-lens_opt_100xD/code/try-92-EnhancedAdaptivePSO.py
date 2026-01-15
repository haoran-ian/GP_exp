import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        num_particles = 15
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])

        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        max_velocity = 0.6
        phase_shift = 0.7
        adaptive_mutation_freq = 10
        inertia_weight = 0.9
        success_threshold = 10
        success_counter = 0

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                phase_factor = np.sin(2 * np.pi * phase_shift * self.evaluations / self.budget)
                inertia = inertia_weight * (1 - (self.evaluations / self.budget)**0.5) * (0.5 + 0.5 * phase_factor)

                neighbor_indices = self._get_diverse_neighbors(particles, idx, num_particles)
                neighbor_best_position = min(neighbor_indices, key=lambda i: personal_best_values[i])

                cognitive = 2.2 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.6 * r2 * (personal_best_positions[neighbor_best_position] - particles[idx])

                velocities[idx] = inertia * velocities[idx] + cognitive + 0.4 * social
                velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if idx == global_best_index:
                    angle = self.evaluations * 0.15 + np.random.normal(0, 0.05)
                    displacement = np.array([
                        0.1 * np.cos(angle),
                        0.1 * np.sin(angle)
                    ] + [0] * (self.dim - 2))
                    spiral_position = global_best_position + displacement[:self.dim]
                    spiral_position = np.clip(spiral_position, func.bounds.lb, func.bounds.ub)
                    particles[idx] = spiral_position

                value = func(particles[idx])
                self.evaluations += 1

                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                    success_counter += 1

                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]
                    inertia_weight = max(0.4, inertia_weight * 0.95)  # Reduce inertia on success
                    success_counter = 0

                if self.evaluations % adaptive_mutation_freq == 0:
                    particles[idx] += np.random.normal(0, 0.03 + 0.1 * (self.evaluations / self.budget), self.dim)
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if success_counter >= success_threshold:
                    inertia_weight = min(0.9, inertia_weight * 1.05)  # Increase inertia if stagnation

                if self.evaluations >= self.budget:
                    break

        return global_best_position, global_best_value

    def _get_diverse_neighbors(self, particles, idx, num_particles):
        neighborhood_size = 4
        base_neighbors = [(idx + i) % num_particles for i in range(-neighborhood_size//2, neighborhood_size//2 + 1)]
        additional_indices = np.random.choice(num_particles, neighborhood_size, replace=False)
        return list(set(base_neighbors + list(additional_indices)))