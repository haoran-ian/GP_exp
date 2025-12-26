import numpy as np

class EnhancedAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 5 * dim
        self.base_cr = 0.9  # Base Crossover probability
        self.base_mutation_factor = 0.8
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb

    def adaptive_parameters(self, iteration, max_iterations):
        # Adaptively adjust crossover and mutation parameters based on progress
        progress = iteration / max_iterations
        cr = self.base_cr - 0.4 * progress  # Decrease crossover probability over time
        mutation_factor = self.base_mutation_factor + 0.2 * (1 - progress)  # Increase mutation factor over time
        return cr, mutation_factor

    def differential_evolution_step(self, bounds, func, iteration, max_iterations):
        new_population = np.copy(self.population)
        cr, mutation_factor = self.adaptive_parameters(iteration, max_iterations)
        for i in range(self.pop_size):
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            mutant_vector = x0 + mutation_factor * (x1 - x2)
            mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < cr
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness < func(self.population[i]):
                new_population[i] = trial_vector
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = trial_fitness
        self.population = new_population

    def stochastic_gradient_local_search(self, bounds, func):
        for i in range(self.pop_size):
            candidate = np.copy(self.population[i])
            gradient = self.estimate_gradient(candidate, func)
            step_size = np.random.uniform(0.005, 0.02) * (bounds.ub - bounds.lb)
            candidate -= step_size * gradient
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
        iterations = 0
        max_iterations = self.budget // self.pop_size
        while evaluations < self.budget:
            self.differential_evolution_step(bounds, func, iterations, max_iterations)
            evaluations += self.pop_size
            iterations += 1
            if evaluations < self.budget:
                self.stochastic_gradient_local_search(bounds, func)
                evaluations += self.pop_size
        return self.best_solution