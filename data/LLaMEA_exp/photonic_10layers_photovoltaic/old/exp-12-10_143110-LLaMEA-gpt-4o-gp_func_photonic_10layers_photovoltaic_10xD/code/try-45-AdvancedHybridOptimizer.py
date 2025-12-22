import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 5 * dim
        self.cr = 0.9  # Initial crossover probability
        self.mutation_factor = 0.8  # Initial mutation factor
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb

    def differential_evolution_step(self, bounds, func):
        new_population = np.copy(self.population)
        for i in range(self.pop_size):
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            mutant_vector = x0 + self.mutation_factor * (x1 - x2)
            mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness < func(self.population[i]):
                new_population[i] = trial_vector
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = trial_fitness
        self.population = new_population

    def adaptive_parameter_tuning(self, iteration, max_iterations):
        # Dynamically adjust mutation factor and crossover probability
        self.mutation_factor = 0.5 + 0.3 * np.sin(np.pi * iteration / max_iterations)
        self.cr = 0.5 + 0.4 * np.cos(np.pi * iteration / max_iterations)

    def stochastic_gradient_descent(self, bounds, func):
        learning_rate = 0.01
        for i in range(self.pop_size):
            candidate = np.copy(self.population[i])
            gradient = self.estimate_gradient(candidate, func)
            candidate -= learning_rate * gradient
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
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
        return gradient

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        evaluations = 0
        max_iterations = self.budget // self.pop_size
        iteration = 0
        
        while evaluations < self.budget:
            self.adaptive_parameter_tuning(iteration, max_iterations)
            self.differential_evolution_step(bounds, func)
            evaluations += self.pop_size
            if evaluations < self.budget:
                self.stochastic_gradient_descent(bounds, func)
                evaluations += self.pop_size
            iteration += 1
        
        return self.best_solution