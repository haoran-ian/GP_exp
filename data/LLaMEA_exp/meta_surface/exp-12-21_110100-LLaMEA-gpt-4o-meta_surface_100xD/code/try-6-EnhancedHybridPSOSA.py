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
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.learning_rate = 0.5  # Added adaptive learning rate

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
            self.inertia_weight = max(self.inertia_weight_min, self.inertia_weight * 0.995)
            
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.velocities = self.inertia_weight * self.velocities \
                + r1 * (self.personal_best - self.particles) \
                + self.learning_rate * r2 * (self.global_best - self.particles)  # Modified
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
                step_size = np.random.normal(0, 0.1, self.dim) * np.random.pareto(1.5, self.dim)
                candidate = self.particles[i] + step_size

                # Differential Evolution Crossover
                idxs = np.random.choice(self.num_particles, 3, replace=False)
                donor = self.personal_best[idxs[0]] + 0.8 * (self.personal_best[idxs[1]] - self.personal_best[idxs[2]])
                crossover = np.where(np.random.rand(self.dim) < 0.9, donor, candidate)
                crossover = np.clip(crossover, bounds[0], bounds[1])

                mutation = np.random.normal(0, 0.01, self.dim)
                candidate = crossover + mutation
                candidate = np.clip(candidate, bounds[0], bounds[1])
                candidate_fitness = func(candidate)
                eval_count += 1
                if candidate_fitness < fitness[i] or np.exp((fitness[i] - candidate_fitness) / self.temp) > np.random.rand():
                    self.particles[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < pbest_fitness[i]:
                        self.personal_best[i] = candidate
                        pbest_fitness[i] = candidate_fitness
                        if candidate_fitness < gbest_fitness:
                            gbest_fitness = candidate_fitness
                            self.global_best = candidate

            self.temp *= self.alpha
            self.alpha = max(0.9 * self.alpha, self.initial_alpha * 0.1)
            self.learning_rate *= 0.99  # Adaptive learning rate decay

        return self.global_best