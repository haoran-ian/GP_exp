import numpy as np

class ImprovedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.cognitive = 2.0
        self.social = 2.0
        self.temperature = 100.0
        self.cooling_rate = 0.95
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim)
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        self.evaluations = self.num_particles
        
        while self.evaluations < self.budget:
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * 
                                          (self.evaluations / self.budget))
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia * velocities[i] +
                                 self.cognitive * r1 * (personal_best[i] - particles[i]) +
                                 self.social * r2 * (global_best - particles[i]))

                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                fitness = func(particles[i])
                self.evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness

                    if np.random.rand() < np.exp((personal_best_values[i] - global_best_value) / self.temperature):
                        global_best = personal_best[i]
                        global_best_value = personal_best_values[i]

            best_particle_index = np.argmin(personal_best_values)
            if personal_best_values[best_particle_index] < global_best_value:
                global_best = personal_best[best_particle_index]
                global_best_value = personal_best_values[best_particle_index]

            self.temperature *= self.cooling_rate

        return global_best