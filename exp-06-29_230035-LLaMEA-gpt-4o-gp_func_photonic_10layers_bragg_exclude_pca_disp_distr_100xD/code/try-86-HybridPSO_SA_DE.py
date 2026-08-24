import numpy as np

class HybridPSO_SA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, 12 * dim)  # Adjusted number of particles
        self.inertia = 0.6  # Adjusted inertia weight
        self.cognitive = 1.7  # Adjusted cognitive component
        self.social = 1.7  # Adjusted social component
        self.temperature = 120.0  # Adjusted initial temperature
        self.cooling_rate = 0.95  # Adjusted cooling rate
        self.mutation_factor = 0.8  # Mutation factor for DE
        self.crossover_rate = 0.9  # Crossover rate for DE

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
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.social * np.random.rand(self.dim) * (global_best - particles[i])
                )
                # Differential Evolution mechanism
                a, b, c = np.random.choice(self.num_particles, 3, replace=False)
                mutant = np.clip(personal_best[a] + self.mutation_factor * (personal_best[b] - personal_best[c]), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                particles[i] = np.where(cross_points, mutant, particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                # Evaluate particle
                fitness = func(particles[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness

                    # Simulated annealing: probabilistically accept worse solutions
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