import numpy as np

class ImprovedDynamicHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 5 * dim
        self.cr = np.linspace(0.5, 0.9, self.pop_size)  # Dynamic crossover probability
        self.mutation_factor = np.linspace(0.5, 0.9, self.pop_size)
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb
        self.elite = np.copy(self.population[0])  # Add elite population initialization

    def differential_evolution_step(self, bounds, func):
        new_population = np.copy(self.population)
        fitness_values = np.array([func(ind) for ind in self.population])
        fitness_variability = np.std(fitness_values)  # Calculate fitness variability
        for i in range(self.pop_size):
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            cr_i = self.cr[i]
            mut_factor_i = self.mutation_factor[i] * (1 + fitness_variability)  # Adapt mutation factor
            mutant_vector = x0 + mut_factor_i * (x1 - x2)
            mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < cr_i
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])
            trial_fitness = func(trial_vector)
            if trial_fitness < func(self.population[i]):
                new_population[i] = trial_vector
                if trial_fitness < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = trial_fitness
        self.population = new_population
        self.population[0] = self.elite  # Apply elitism by retaining best solution

    def stochastic_gradient_based_local_search(self, bounds, func):
        for i in range(self.pop_size):
            candidate = np.copy(self.population[i])
            gradient = self.estimate_gradient(candidate, func)
            stochastic_step = np.random.uniform(0.005, 0.02) * (bounds.ub - bounds.lb)
            candidate -= stochastic_step * gradient
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
            evaluations += self.pop_size
            if evaluations < self.budget:
                self.stochastic_gradient_based_local_search(bounds, func)
                evaluations += self.pop_size
        return self.best_solution