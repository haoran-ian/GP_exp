import numpy as np

class AdaptiveDifferentialPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, budget // 2)
        self.particles = np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.rand(self.num_particles, self.dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = self.particles[np.random.choice(range(self.num_particles))]
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9

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
                + self.c1 * r1 * (self.personal_best - self.particles) \
                + self.c2 * r2 * (self.global_best - self.particles)

            # Adaptation based on diversity
            population_std = np.std(self.particles, axis=0)
            diversity_factor = np.mean(population_std) / np.max([1e-10, np.linalg.norm(self.global_best)])
            self.mutation_factor = 0.5 + 0.5 * diversity_factor

            # Differential mutation
            for i in range(self.num_particles):
                indices = np.random.choice(self.num_particles, 3, replace=False)
                a, b, c = self.particles[indices[0]], self.particles[indices[1]], self.particles[indices[2]]
                mutant_vector = a + self.mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, bounds[0], bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(cross_points, mutant_vector, self.particles[i])
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

            self.particles = np.clip(self.particles + self.velocities, bounds[0], bounds[1])
            fitness = np.apply_along_axis(func, 1, self.particles)
            eval_count += self.num_particles

            improved = fitness < pbest_fitness
            self.personal_best[improved] = self.particles[improved]
            pbest_fitness[improved] = fitness[improved]

            if np.min(fitness) < gbest_fitness:
                gbest_fitness = np.min(fitness)
                self.global_best = self.particles[np.argmin(fitness)]

        return self.global_best