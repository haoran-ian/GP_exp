import numpy as np

class EnhancedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(40, 10 * dim)  # A reasonable number of particles
        self.inertia = 0.7  # Inertia weight
        self.cognitive = 1.5  # Cognitive component
        self.social = 1.5  # Social component
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.99  # Cooling rate for temperature
        self.f = 0.8  # Differential evolution mutation factor
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim) * 0.1  # Small initial velocities
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles

        while evaluations < self.budget:
            # Adaptive Differential Evolution
            for i in range(self.num_particles):
                indices = list(range(self.num_particles))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(particles[a] + self.f * (particles[b] - particles[c]), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, particles[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_fitness

                    # Simulated annealing acceptance
                    if np.random.rand() < np.exp((personal_best_values[i] - global_best_value) / self.temperature):
                        global_best = personal_best[i]
                        global_best_value = personal_best_values[i]

            # Particle Swarm Optimization
            for i in range(self.num_particles):
                velocities[i] = (
                    self.inertia * velocities[i]
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
                        global_best = particles[i]
                        global_best_value = fitness

            # Cooling the temperature
            self.temperature *= self.cooling_rate

        return global_best