import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)
        self.initial_inertia = 0.9
        self.final_inertia = 0.4
        self.cognitive = 1.5
        self.social = 1.5
        self.temperature = 100.0
        self.min_temperature = 1e-3
        self.cooling_rate = 0.99

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim) * (bounds[1] - bounds[0])
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles
        step = 0

        while evaluations < self.budget:
            inertia_weight = self.initial_inertia - (self.initial_inertia - self.final_inertia) * (evaluations / self.budget)
            for i in range(self.num_particles):
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + self.cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.social * np.random.rand(self.dim) * (global_best - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])
                fitness = func(particles[i])
                evaluations += 1
                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness
                    if fitness < global_best_value:
                        global_best = personal_best[i]
                        global_best_value = fitness
                else:
                    if np.random.rand() < np.exp((fitness - global_best_value) / max(self.temperature, self.min_temperature)):
                        global_best = particles[i]
                        global_best_value = fitness

            if evaluations % (self.num_particles * 10) == 0:
                self.temperature *= self.cooling_rate

        return global_best