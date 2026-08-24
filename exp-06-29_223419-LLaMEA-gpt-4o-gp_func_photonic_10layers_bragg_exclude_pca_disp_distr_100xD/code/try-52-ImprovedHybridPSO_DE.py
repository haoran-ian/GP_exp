import numpy as np

class ImprovedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.global_best = None
        self.func_evals = 0

    def initialize(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.global_best = self.population[np.random.choice(self.population_size)]

    def evaluate(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.func_evals += len(fitness)
        return fitness

    def adapt_parameters(self, iteration, max_iterations):
        inertia = 0.9 - 0.5 * (iteration / max_iterations)
        cognitive = 1.5 + 0.5 * (iteration / max_iterations)
        social = 2.0 - 0.5 * (iteration / max_iterations)
        F = 0.8 - 0.3 * (iteration / max_iterations)
        return inertia, cognitive, social, F

    def local_search(self, candidate, bounds, func):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        new_candidate = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
        new_fitness = func(new_candidate)
        self.func_evals += 1
        return new_candidate, new_fitness

    def optimize(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        fitness = self.evaluate(func)
        personal_best_fitness = fitness.copy()
        global_best_fitness = np.min(fitness)
        
        max_iterations = self.budget // self.population_size
        for iteration in range(max_iterations):
            # Update personal and global best
            better_mask = fitness < personal_best_fitness
            self.personal_best[better_mask] = self.population[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]
            if np.min(fitness) < global_best_fitness:
                self.global_best = self.population[np.argmin(fitness)]
                global_best_fitness = np.min(fitness)
            
            # Adaptive PSO update
            inertia, cognitive, social, F = self.adapt_parameters(iteration, max_iterations)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocities = (inertia * self.velocities +
                               cognitive * r1 * (self.personal_best - self.population) +
                               social * r2 * (self.global_best - self.population))
            self.population += self.velocities
            np.clip(self.population, bounds.lb, bounds.ub, out=self.population)

            # Adaptive Differential Evolution update
            CR = 0.9  # Constant crossover probability
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial_vector = np.where(cross_points, mutant_vector, self.population[i])
                trial_fitness = func(trial_vector)
                self.func_evals += 1
                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness
                else:
                    # Apply local search to escape local optima
                    new_vector, new_fitness = self.local_search(self.population[i], bounds, func)
                    if new_fitness < fitness[i]:
                        self.population[i] = new_vector
                        fitness[i] = new_fitness

            if self.func_evals >= self.budget:
                break

    def __call__(self, func):
        self.optimize(func)
        return self.global_best