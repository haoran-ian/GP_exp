import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)
        self.inertia = 0.9  # Start with higher inertia for exploration
        self.cognitive = 2.0  # Higher cognitive component for stronger personal learning
        self.social = 2.0  # Higher social component for stronger collaboration
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995  # Slower cooling rate
        self.temperature = self.initial_temperature

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim) * (bounds[1] - bounds[0]) * 0.1
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Dynamically adjust inertia weight
                self.inertia = 0.9 - (0.5 * evaluations / self.budget)

                # Update velocity with momentum
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

                    # Simulated annealing: probabilistically accept worse solutions
                    prob_accept = np.exp((personal_best_values[i] - global_best_value) / self.temperature)
                    if fitness < global_best_value or np.random.rand() < prob_accept:
                        global_best = personal_best[i]
                        global_best_value = personal_best_values[i]

            # Update global best and adjust temperature
            best_particle_index = np.argmin(personal_best_values)
            if personal_best_values[best_particle_index] < global_best_value:
                global_best = personal_best[best_particle_index]
                global_best_value = personal_best_values[best_particle_index]

            # Cool the temperature
            self.temperature *= self.cooling_rate

        return global_best