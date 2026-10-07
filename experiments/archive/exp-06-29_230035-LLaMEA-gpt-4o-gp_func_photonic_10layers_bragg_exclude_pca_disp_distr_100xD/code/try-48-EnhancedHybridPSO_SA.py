import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.cognitive = 1.5
        self.social = 1.5
        self.initial_temperature = 100.0
        self.cooling_rate_max = 0.99
        self.cooling_rate_min = 0.9
        self.evaluations = 0

    def adaptive_inertia(self):
        return self.inertia_max - (self.inertia_max - self.inertia_min) * (self.evaluations / self.budget)

    def dynamic_cooling(self):
        return self.cooling_rate_min + (self.cooling_rate_max - self.cooling_rate_min) * (1 - self.evaluations / self.budget)

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim)
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        self.evaluations = self.num_particles
        temperature = self.initial_temperature

        while self.evaluations < self.budget:
            inertia = self.adaptive_inertia()
            cooling_rate = self.dynamic_cooling()
            
            for i in range(self.num_particles):
                # Update velocity
                velocities[i] = (
                    inertia * velocities[i]
                    + self.cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.social * np.random.rand(self.dim) * (global_best - particles[i])
                )
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                # Evaluate particle
                fitness = func(particles[i])
                self.evaluations += 1

                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness

                    # Simulated annealing
                    if np.random.rand() < np.exp((personal_best_values[i] - global_best_value) / temperature):
                        global_best = personal_best[i]
                        global_best_value = personal_best_values[i]

            # Update global best
            best_particle_index = np.argmin(personal_best_values)
            if personal_best_values[best_particle_index] < global_best_value:
                global_best = personal_best[best_particle_index]
                global_best_value = personal_best_values[best_particle_index]

            # Cooling the temperature
            temperature *= cooling_rate

        return global_best