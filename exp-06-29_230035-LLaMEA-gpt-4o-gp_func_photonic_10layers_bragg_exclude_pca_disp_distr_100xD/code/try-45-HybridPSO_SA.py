import numpy as np

class HybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)  # A reasonable number of particles
        self.inertia = 0.9  # Enhanced: start with a higher inertia weight
        self.cognitive = 1.5  # Cognitive component
        self.social = 1.5  # Social component
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.95  # Enhanced: slower cooling rate for better exploration

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
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)  # Add randomness factors
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * r1 * (personal_best[i] - particles[i])
                    + self.social * r2 * (global_best - particles[i])
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

            # Adaptive inertia weight update
            self.inertia = 0.4 + (0.5 * (self.budget - evaluations) / self.budget)

        return global_best