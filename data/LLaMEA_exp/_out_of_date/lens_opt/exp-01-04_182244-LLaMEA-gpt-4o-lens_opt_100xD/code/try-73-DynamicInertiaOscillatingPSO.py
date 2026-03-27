import numpy as np

class DynamicInertiaOscillatingPSO:
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
        adaptive_mutation_freq = 10

        while self.evaluations < self.budget:
            for idx in range(num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # Dynamic inertia weight oscillating between 0.4 and 0.9
                inertia = 0.4 + 0.5 * np.sin(2 * np.pi * self.evaluations / self.budget)
                
                # Dynamic acceleration coefficients
                c1 = 1.5 + 1.0 * (self.evaluations / self.budget)
                c2 = 2.5 - 1.0 * (self.evaluations / self.budget)
                
                cognitive = c1 * r1 * (personal_best_positions[idx] - particles[idx])
                social = c2 * r2 * (global_best_position - particles[idx])
                
                velocities[idx] = inertia * velocities[idx] + cognitive + social
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

                if self.evaluations % adaptive_mutation_freq == 0:
                    particles[idx] += np.random.normal(0, 0.03 + 0.1 * (self.evaluations / self.budget), self.dim)
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                if self.evaluations >= self.budget:
                    break

        return global_best_position, global_best_value