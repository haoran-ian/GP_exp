import numpy as np

class EnhancedAdaptivePSO:
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
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.inertia_weight_max = 1.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.topology_switch_threshold = 0.5

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        self.particles = bounds[0] + self.particles * (bounds[1] - bounds[0])
        self.personal_best = self.particles.copy()
        fitness = np.apply_along_axis(func, 1, self.particles)
        pbest_fitness = fitness.copy()
        gbest_fitness = np.min(fitness)
        self.global_best = self.particles[np.argmin(fitness)]

        eval_count = self.num_particles
        topology_switch_counter = 0

        while eval_count < self.budget:
            self.inertia_weight = max(
                self.inertia_weight_min,
                self.inertia_weight_max - (eval_count / self.budget) * (self.inertia_weight_max - self.inertia_weight_min)
            )
            
            if topology_switch_counter % 20 < 10:
                local_best = self.global_best
            else:
                local_best_indices = np.roll(np.arange(self.num_particles), -1)
                local_best = self.particles[local_best_indices[np.argmin(fitness[local_best_indices])]]

            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.velocities = self.inertia_weight * self.velocities \
                + self.c1 * r1 * (self.personal_best - self.particles) \
                + self.c2 * r2 * (local_best - self.particles)
            self.particles = np.clip(self.particles + self.velocities, bounds[0], bounds[1])

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
                candidate = np.clip(self.particles[i] + step_size, bounds[0], bounds[1])
                mutation = np.random.normal(0, 0.01, self.dim)
                candidate = np.clip(candidate + mutation, bounds[0], bounds[1])
                candidate_fitness = func(candidate)
                eval_count += 1
                if candidate_fitness < fitness[i]:
                    self.particles[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < pbest_fitness[i]:
                        self.personal_best[i] = candidate
                        pbest_fitness[i] = candidate_fitness
                        if candidate_fitness < gbest_fitness:
                            gbest_fitness = candidate_fitness
                            self.global_best = candidate

            topology_switch_counter += 1

        return self.global_best