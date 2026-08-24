import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)
        self.inertia = 0.9  # Increased inertia for broader exploration at the start
        self.cognitive = 2.0  # Enhanced cognitive component for better personal learning
        self.social = 2.0  # Enhanced social component for better global learning
        self.temperature = 100.0
        self.cooling_rate = 0.95  # Slightly faster cooling for quicker convergence
        self.vel_max = 0.1 * (func.bounds.ub - func.bounds.lb)  # Velocity clamping

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Adaptive inertia
                adapt_inertia = self.inertia - (self.inertia * evaluations / self.budget)

                # Update velocity with adaptive inertia
                velocities[i] = (
                    adapt_inertia * velocities[i]
                    + self.cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.social * np.random.rand(self.dim) * (global_best - particles[i])
                )
                velocities[i] = np.clip(velocities[i], -self.vel_max, self.vel_max)

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