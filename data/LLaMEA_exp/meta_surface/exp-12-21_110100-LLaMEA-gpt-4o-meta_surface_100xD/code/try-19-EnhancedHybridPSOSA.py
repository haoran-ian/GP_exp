import numpy as np

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, budget // 2)
        self.particles = np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.rand(self.num_particles, self.dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = self.particles[np.random.choice(range(self.num_particles))]
        self.temp = 100
        self.alpha = 0.99
        self.initial_alpha = self.alpha
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.inertia_weight = self.inertia_weight_max
        self.learning_factor = 2.0
        self.F = 0.5  # Differential weight

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        self.particles = bounds[0] + self.particles * (bounds[1] - bounds[0])
        self.personal_best = self.particles.copy()
        fitness = np.apply_along_axis(func, 1, self.particles)
        pbest_fitness = fitness.copy()
        gbest_fitness = np.min(fitness)
        self.global_best = self.particles[np.argmin(fitness)]

        eval_count = self.num_particles

        while eval_count < self.budget:
            self.inertia_weight = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * (
                        1 - eval_count / self.budget) ** 2
            
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.velocities = self.inertia_weight * self.velocities \
                + self.learning_factor * r1 * (self.personal_best - self.particles) \
                + self.learning_factor * r2 * (self.global_best - self.particles)
            self.particles = self.particles + self.velocities
            self.particles = np.clip(self.particles, bounds[0], bounds[1])

            fitness = np.apply_along_axis(func, 1, self.particles)
            eval_count += self.num_particles

            improved = fitness < pbest_fitness
            self.personal_best[improved] = self.particles[improved]
            pbest_fitness[improved] = fitness[improved]

            if np.min(fitness) < gbest_fitness:
                gbest_fitness = np.min(fitness)
                self.global_best = self.particles[np.argmin(fitness)]

            for i in range(self.num_particles):
                indices = np.random.choice(range(self.num_particles), 3, replace=False)
                x1, x2, x3 = self.particles[indices]
                mutant_vector = x1 + self.F * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < 0.9
                trial_vector = np.where(crossover, mutant_vector, self.particles[i])
                
                mutation = np.random.normal(0, 0.01, self.dim)
                trial_vector += mutation
                trial_vector = np.clip(trial_vector, bounds[0], bounds[1])
                
                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    self.particles[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < pbest_fitness[i]:
                        self.personal_best[i] = trial_vector
                        pbest_fitness[i] = trial_fitness
                        if trial_fitness < gbest_fitness:
                            gbest_fitness = trial_fitness
                            self.global_best = trial_vector

            self.temp *= self.alpha
            self.alpha = max(0.9 * self.alpha, self.initial_alpha * 0.1)

        return self.global_best