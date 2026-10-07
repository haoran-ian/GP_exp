import numpy as np

class EnhancedHybridPSO_DE:
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
        self.global_best = self.population[np.argmin(self.evaluate_population(self.population))]
    
    def evaluate_population(self, population):
        fitness = np.apply_along_axis(func, 1, population)
        self.func_evals += len(fitness)
        return fitness

    def update_best(self, fitness, personal_best_fitness):
        better_mask = fitness < personal_best_fitness
        self.personal_best[better_mask] = self.population[better_mask]
        personal_best_fitness[better_mask] = fitness[better_mask]
        if np.min(fitness) < np.min(personal_best_fitness):
            self.global_best = self.population[np.argmin(fitness)]
    
    def optimize(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        fitness = self.evaluate_population(self.population)
        personal_best_fitness = fitness.copy()
        
        for _ in range(self.budget // self.population_size):
            # Update personal and global best with elitism
            self.update_best(fitness, personal_best_fitness)
            
            # Adaptive parameters
            inertia = 0.5 + np.random.rand() * 0.1
            cognitive = 1.5 + np.random.rand() * 0.1
            social = 2.0 + np.random.rand() * 0.1
            
            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocities = (inertia * self.velocities +
                               cognitive * r1 * (self.personal_best - self.population) +
                               social * r2 * (self.global_best - self.population))
            self.population += self.velocities
            np.clip(self.population, bounds.lb, bounds.ub, out=self.population)

            # Differential Evolution update
            F = 0.8  # Differential weight
            CR = 0.9  # Crossover probability
            new_population = self.population.copy()
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
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
            self.population = new_population

            if self.func_evals >= self.budget:
                break

    def __call__(self, func):
        self.optimize(func)
        return self.global_best