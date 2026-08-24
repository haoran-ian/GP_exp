import numpy as np

class RefinedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, 10 * dim)  # Increased number of particles for better coverage
        self.inertia = 0.9  # Starting inertia weight
        self.cognitive = 2.0  # Enhanced cognitive component to improve personal exploration
        self.social = 2.0  # Enhanced social component for better global convergence
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.95  # Reduced cooling rate for prolonged exploration
        self.inertia_damping = 0.99  # Damping factor for inertia weight reduction

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim) * (bounds[1] - bounds[0]) / 2
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocity
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
                    if np.random.rand() < np.exp((fitness - global_best_value) / self.temperature):
                        global_best = personal_best[i]
                        global_best_value = fitness

            # Update global best
            best_particle_index = np.argmin(personal_best_values)
            if personal_best_values[best_particle_index] < global_best_value:
                global_best = personal_best[best_particle_index]
                global_best_value = personal_best_values[best_particle_index]

            # Cooling the temperature
            self.temperature *= self.cooling_rate
            # Inertia damping
            self.inertia *= self.inertia_damping

        return global_best