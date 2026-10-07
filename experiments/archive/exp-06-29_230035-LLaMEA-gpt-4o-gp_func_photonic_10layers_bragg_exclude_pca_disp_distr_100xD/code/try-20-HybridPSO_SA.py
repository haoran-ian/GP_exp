import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)  # A reasonable number of particles
        self.inertia = 0.7  # Inertia weight
        self.cognitive = 1.5  # Cognitive component
        self.social = 1.5  # Social component
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99  # Cooling rate for temperature

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim)
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocity with adaptive cognitive and social components
                self.cognitive = self.cognitive * 0.9 if np.random.rand() > 0.5 else self.cognitive * 1.1
                self.social = self.social * 0.9 if np.random.rand() > 0.5 else self.social * 1.1
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.social * np.random.rand(self.dim) * (global_best - particles[i])
                )
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                # Evaluate particle
                fitness = func(particles[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness

                    # Simulated annealing: probabilistically accept worse solutions to escape local minima
                    if np.random.rand() < np.exp((personal_best_values[i] - global_best_value) / self.temperature):
                        global_best = personal_best[i]
                        global_best_value = personal_best_values[i]

            # Update global best
            best_particle_index = np.argmin(personal_best_values)
            if personal_best_values[best_particle_index] < global_best_value:
                global_best = personal_best[best_particle_index]
                global_best_value = personal_best_values[best_particle_index]

            # Cooling the temperature
            self.temperature *= self.cooling_rate

        return global_best