import numpy as np

class EnhancedAdaptiveDiversityEnhancedPSO:
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
        angle_increment = 0.15
        initial_radius = 0.1
        radius = initial_radius
        max_velocity = 0.6
        phase_shift = 0.7
        adaptive_mutation_freq = 10
        learning_rate = 0.01

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                phase_factor = np.sin(2 * np.pi * phase_shift * self.evaluations / self.budget)
                inertia = 0.9 * (1 - (self.evaluations / self.budget)**0.5) * (0.5 + 0.5 * phase_factor)
                
                # Adaptive neighborhood diversity
                neighbor_indices = self._get_diverse_neighbors(particles, idx, num_particles)
                neighbor_best_position = min(neighbor_indices, key=lambda i: personal_best_values[i])
                
                cognitive = 2.1 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.6 * r2 * (personal_best_positions[neighbor_best_position] - particles[idx])
                
                velocities[idx] = inertia * velocities[idx] + cognitive + 0.4 * social
                velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                # Gradient-informed perturbation
                if idx == global_best_index:
                    gradient = self._estimate_gradient(func, global_best_position)
                    perturbation_direction = gradient / (np.linalg.norm(gradient) + 1e-8)
                    particles[idx] += learning_rate * perturbation_direction

                # Evaluation and update
                value = func(particles[idx])
                self.evaluations += 1

                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]
                    radius = max(initial_radius, radius * 0.8)

                # Adaptive mutation
                if self.evaluations % adaptive_mutation_freq == 0:
                    particles[idx] += np.random.normal(0, 0.03 + 0.1 * (self.evaluations / self.budget), self.dim)
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if self.evaluations >= self.budget:
                    break
            
            if all(personal_best_values >= global_best_value):
                radius *= 1.1

        return global_best_position, global_best_value

    def _get_diverse_neighbors(self, particles, idx, num_particles):
        neighborhood_size = 4
        base_neighbors = [(idx + i) % num_particles for i in range(-neighborhood_size//2, neighborhood_size//2 + 1)]
        additional_indices = np.random.choice(num_particles, neighborhood_size, replace=False)
        return list(set(base_neighbors + list(additional_indices)))

    def _estimate_gradient(self, func, position):
        epsilon = 1e-5
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturbed_position = position.copy()
            perturbed_position[i] += epsilon
            gradient[i] = (func(perturbed_position) - func(position)) / epsilon
        return gradient