import numpy as np

class AdaptiveQuantumHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.global_best = None
        self.func_evals = 0
        # Scaling factors for adaptation
        self.inertia_weight = 0.9
        self.cognitive_weight = 2.0
        self.social_weight = 2.0

    def initialize(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.global_best = self.population[np.random.choice(self.population_size)]

    def evaluate(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.func_evals += len(fitness)
        return fitness

    def adapt_weights(self, iter, max_iter):
        # Linearly decreasing inertia weight
        self.inertia_weight = 0.9 - (0.5 * (iter / max_iter))
        # Dynamic adjustment of cognitive and social weights
        self.cognitive_weight = 2.5 - (1.5 * (iter / max_iter))
        self.social_weight = 0.5 + (1.5 * (iter / max_iter))

    def quantum_inspired_update(self, global_best, bounds):
        # Quantum behavior inspired by potential wells
        center = (self.population + global_best) / 2.0
        spread = np.abs(self.population - global_best)
        self.population = np.clip(center + spread * (np.random.rand(self.population_size, self.dim) - 0.5),
                                  bounds.lb, bounds.ub)

    def optimize(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        fitness = self.evaluate(func)
        personal_best_fitness = fitness.copy()
        global_best_fitness = np.min(fitness)
        max_iter = self.budget // self.population_size
        
        for iter in range(max_iter):
            # Update personal and global best
            better_mask = fitness < personal_best_fitness
            self.personal_best[better_mask] = self.population[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]
            if np.min(fitness) < global_best_fitness:
                self.global_best = self.population[np.argmin(fitness)]
                global_best_fitness = np.min(fitness)

            self.adapt_weights(iter, max_iter)

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocities = (self.inertia_weight * self.velocities +
                               self.cognitive_weight * r1 * (self.personal_best - self.population) +
                               self.social_weight * r2 * (self.global_best - self.population))
            self.population += self.velocities
            np.clip(self.population, bounds.lb, bounds.ub, out=self.population)

            # Quantum-inspired update
            self.quantum_inspired_update(self.global_best, bounds)

            fitness = self.evaluate(func)

            if self.func_evals >= self.budget:
                break

    def __call__(self, func):
        self.optimize(func)
        return self.global_best