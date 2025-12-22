import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 5 * dim
        self.cr = 0.9  # Crossover probability
        self.mutation_factor = 0.8
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
            if func(trial_vector) < func(self.population[i]):
                new_population[i] = trial_vector
                if func(trial_vector) < self.best_fitness:
                    self.best_solution = trial_vector
                    self.best_fitness = func(trial_vector)
        self.population = new_population

    def local_search(self, bounds, func):
        step_size = 0.1 * (bounds.ub - bounds.lb)
        for i in range(self.pop_size):
            for j in range(self.dim):
                candidate = np.copy(self.population[i])
                candidate[j] += step_size[j] * np.random.uniform(-1, 1)
                candidate = np.clip(candidate, bounds.lb, bounds.ub)
                if func(candidate) < func(self.population[i]):
                    self.population[i] = candidate
                    if func(candidate) < self.best_fitness:
                        self.best_solution = candidate
                        self.best_fitness = func(candidate)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        evaluations = 0
        while evaluations < self.budget:
            self.differential_evolution_step(bounds, func)
            evaluations += self.pop_size
            if evaluations < self.budget:
                self.local_search(bounds, func)
                evaluations += self.pop_size
            # Adaptive crossover probability adjustment
            self.cr = 0.9 * (1 - (evaluations / self.budget)) + 0.1
        return self.best_solution