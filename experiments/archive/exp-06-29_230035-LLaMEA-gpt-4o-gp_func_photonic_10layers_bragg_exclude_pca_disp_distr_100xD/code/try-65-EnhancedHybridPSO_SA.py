import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)
        self.min_inertia = 0.4
        self.max_inertia = 0.9
        self.cognitive = 1.5
        self.social = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.adaptive_cooling = True

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim)
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles
        iter_count = 0

        while evaluations < self.budget:
            inertia = self.max_inertia - ((self.max_inertia - self.min_inertia) * evaluations / self.budget)
            
            for i in range(self.num_particles):
                velocities[i] = (
                    inertia * velocities[i]
                    + self.cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.social * np.random.rand(self.dim) * (global_best - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                fitness = func(particles[i])
                evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness

                if fitness < global_best_value or (np.random.rand() < np.exp((fitness - global_best_value) / self.temperature)):
                    global_best = particles[i]
                    global_best_value = fitness

            best_particle_index = np.argmin(personal_best_values)
            if personal_best_values[best_particle_index] < global_best_value:
                global_best = personal_best[best_particle_index]
                global_best_value = personal_best_values[best_particle_index]

            if self.adaptive_cooling:
                self.temperature *= np.exp(-iter_count / self.budget)
            else:
                self.temperature *= self.cooling_rate

            iter_count += 1

        return global_best