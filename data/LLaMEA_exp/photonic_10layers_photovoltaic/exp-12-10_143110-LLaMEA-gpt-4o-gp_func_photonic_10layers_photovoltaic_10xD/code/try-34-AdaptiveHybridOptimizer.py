import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 5 * dim
        self.cr = 0.9  # Crossover probability
        self.mutation_factor = 0.8
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb

    def adaptive_mutation_factor(self):
        return self.mutation_factor * (1 - self.evaluations / self.budget)

    def differential_evolution_step(self, bounds, func):
        new_population = np.copy(self.population)
        mutation_factor = self.adaptive_mutation_factor()
        for i in range(self.pop_size):
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            mutant_vector = x0 + mutation_factor * (x1 - x2)
            mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            self.evaluations += 1
            if trial_fitness < func(self.population[i]):
                new_population[i] = trial_vector
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = trial_fitness
        self.population = new_population

    def stochastic_gradient_descent(self, bounds, func):
        for i in range(self.pop_size):
            candidate = np.copy(self.population[i])
            gradient = self.estimate_gradient(candidate, func)
            step_size = 0.01 * (bounds.ub - bounds.lb)
            candidate -= step_size * gradient * np.random.rand(self.dim)
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            self.evaluations += 1
            if candidate_fitness < func(self.population[i]):
                self.population[i] = candidate
                if candidate_fitness < self.best_fitness:
                    self.best_solution = candidate
                    self.best_fitness = candidate_fitness

    def estimate_gradient(self, x, func, epsilon=1e-8):
        gradient = np.zeros_like(x)
        fx = func(x)
        for i in range(self.dim):
            x_step = np.copy(x)
            x_step[i] += epsilon
            gradient[i] = (func(x_step) - fx) / epsilon
            self.evaluations += 1
        return gradient

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        while self.evaluations < self.budget:
            self.differential_evolution_step(bounds, func)
            if self.evaluations < self.budget:
                self.stochastic_gradient_descent(bounds, func)
        return self.best_solution