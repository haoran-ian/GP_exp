import numpy as np

class AdaptiveSpiralParticleSwarmLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def levy_flight(self, size, alpha=1.5):
        # Generate Lévy flights using the Mantegna's algorithm
        sigma_u = (np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
                   (np.math.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v)**(1 / alpha)

    def __call__(self, func):
        # Initialize parameters
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
        
        while self.evaluations < self.budget:
            for idx in range(num_particles):
                # Update velocity and position using PSO formula
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = 0.7
                cognitive = 2.0 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.5 * r2 * (global_best_position - particles[idx])
                velocities[idx] = inertia * velocities[idx] + cognitive + social
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                # Spiral exploration locally around global_best_position
                if idx == global_best_index:
                    angle = self.evaluations * angle_increment
                    displacement = np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle)
                    ] + [0] * (self.dim - 2))
                    spiral_position = global_best_position + displacement[:self.dim]
                    spiral_position = np.clip(spiral_position, func.bounds.lb, func.bounds.ub)
                    particles[idx] = spiral_position

                # Evaluate particles
                value = func(particles[idx])
                self.evaluations += 1

                # Update personal and global bests
                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]
                    radius = max(initial_radius, radius * 0.8)  # Adaptive radius reduction

                # Introduce Lévy flight-based perturbation if no improvement
                if self.evaluations % 20 == 0 or all(personal_best_values >= global_best_value):
                    levy_step = self.levy_flight(self.dim)
                    particles[idx] += levy_step
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)
                    radius *= 1.2  # Increase radius if no improvement

                if self.evaluations >= self.budget:
                    break

        return global_best_position, global_best_value