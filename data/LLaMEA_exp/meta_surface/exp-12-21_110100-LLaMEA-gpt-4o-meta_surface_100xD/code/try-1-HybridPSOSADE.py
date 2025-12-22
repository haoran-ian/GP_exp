import numpy as np

class HybridPSOSADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, budget // 2)  # Dynamic adjustment based on budget
        self.particles = np.random.rand(self.num_particles, self.dim)
        self.velocities = np.random.rand(self.num_particles, self.dim) * 0.1
        self.personal_best = self.particles.copy()
        self.global_best = self.particles[np.random.choice(range(self.num_particles))]
        self.temp = 100  # Initial temperature for Simulated Annealing
        self.alpha = 0.99  # Cooling rate
        self.cross_prob = 0.7  # Crossover probability for DE
        self.f_weight = 0.5  # Differential weight for DE

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
            # PSO Update
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities = 0.5 * self.velocities \
                + r1 * (self.personal_best - self.particles) \
                + r2 * (self.global_best - self.particles)
            self.particles = self.particles + self.velocities
            self.particles = np.clip(self.particles, bounds[0], bounds[1])

            # Evaluate particles
            fitness = np.apply_along_axis(func, 1, self.particles)
            eval_count += self.num_particles

            # Update personal and global best
            improved = fitness < pbest_fitness
            self.personal_best[improved] = self.particles[improved]
            pbest_fitness[improved] = fitness[improved]

            if np.min(fitness) < gbest_fitness:
                gbest_fitness = np.min(fitness)
                self.global_best = self.particles[np.argmin(fitness)]

            # Simulated Annealing local search
            for i in range(self.num_particles):
                candidate = self.particles[i] + np.random.normal(0, 0.1, self.dim)
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

            # Differential Evolution Mutation and Crossover
            for i in range(self.num_particles):
                idxs = [idx for idx in range(self.num_particles) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.f_weight * (b - c)
                mutant = np.clip(mutant, bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.cross_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.particles[i])
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    self.particles[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < pbest_fitness[i]:
                        self.personal_best[i] = trial
                        pbest_fitness[i] = trial_fitness
                        if trial_fitness < gbest_fitness:
                            gbest_fitness = trial_fitness
                            self.global_best = trial

            # Cool down temperature
            self.temp *= self.alpha

        return self.global_best