import numpy as np

class HybridAdaptiveSpiralLevyFlightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / beta)
        return step

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
        
        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = 0.7
                cognitive = 2.0 * r1 * (personal_best_positions[idx] - particles[idx])
                social = 1.5 * r2 * (global_best_position - particles[idx])
                velocities[idx] = inertia * velocities[idx] + cognitive + social
                particles[idx] += velocities[idx]
                particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                # Spiral and Lévy flight exploration
                if idx == global_best_index:
                    angle = self.evaluations * angle_increment
                    displacement = np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle)
                    ] + [0] * (self.dim - 2))
                    spiral_position = global_best_position + displacement[:self.dim]
                    spiral_position = np.clip(spiral_position, func.bounds.lb, func.bounds.ub)
                    particles[idx] = spiral_position
                
                if self.evaluations % 5 == 0:
                    levy_step = self.levy_flight(self.dim)
                    particles[idx] += levy_step
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                value = func(particles[idx])
                self.evaluations += 1

                if value < personal_best_values[idx]:
                    personal_best_values[idx] = value
                    personal_best_positions[idx] = particles[idx]
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particles[idx]
                    radius = max(initial_radius, radius * 0.8)

                if self.evaluations >= self.budget:
                    break
            
            if all(personal_best_values >= global_best_value):
                radius *= 1.2

        return global_best_position, global_best_value