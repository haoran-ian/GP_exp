import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_pop_size = 10 + 5 * dim
        self.cr = 0.9  # Crossover probability
        self.mutation_factor = 0.8
        self.best_solution = None
        self.best_fitness = float('inf')
        self.momentum = 0.9  # Momentum for local search
        self.velocity = np.zeros(self.dim)  # Initialize velocity for momentum-based local search
        
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop_size = self.adaptive_pop_size()
        self.population = np.random.rand(pop_size, self.dim) * (ub - lb) + lb

    def adaptive_pop_size(self):
        # Adaptive population size based on remaining budget
        remaining_budget_ratio = self.budget / (self.init_pop_size * self.dim)
        return max(5, int(self.init_pop_size * remaining_budget_ratio))

    def differential_evolution_step(self, bounds, func):
        new_population = np.copy(self.population)
        for i in range(len(self.population)):
            indices = np.random.choice(range(len(self.population)), 3, replace=False)
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

    def momentum_based_local_search(self, bounds, func):
        for i in range(len(self.population)):
            candidate = np.copy(self.population[i])
            gradient = self.estimate_gradient(candidate, func)
            self.velocity = self.momentum * self.velocity - 0.01 * gradient
            candidate += self.velocity
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
        while evaluations < self.budget:
            self.differential_evolution_step(bounds, func)
            evaluations += len(self.population)
            if evaluations < self.budget:
                self.momentum_based_local_search(bounds, func)
                evaluations += len(self.population)
        return self.best_solution