import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)
        self.inertia = 0.9
        self.cognitive = 2.0
        self.social = 2.0
        self.initial_temperature = 100.0
        self.cooling_rate = 0.95
        self.min_temperature = 1e-3

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles
        temperature = self.initial_temperature

        while evaluations < self.budget and temperature > self.min_temperature:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * r1 * (personal_best[i] - particles[i])
                    + self.social * r2 * (global_best - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                fitness = func(particles[i])
                evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness

                if fitness < global_best_value or np.random.rand() < np.exp((fitness - global_best_value) / temperature):
                    global_best = particles[i]
                    global_best_value = fitness

            best_particle_index = np.argmin(personal_best_values)
            if personal_best_values[best_particle_index] < global_best_value:
                global_best = personal_best[best_particle_index]
                global_best_value = personal_best_values[best_particle_index]

            temperature *= self.cooling_rate
            self.inertia = max(0.4, self.inertia * 0.99)

        return global_best