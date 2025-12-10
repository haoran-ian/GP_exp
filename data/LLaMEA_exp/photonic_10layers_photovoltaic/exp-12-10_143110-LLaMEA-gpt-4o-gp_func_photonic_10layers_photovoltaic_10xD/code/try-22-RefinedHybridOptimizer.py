import numpy as np

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 5 * dim
        self.cr = 0.9  # Crossover probability
        self.base_mutation_factor = 0.5
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb

    def adaptive_mutation_factor(self, func_evals):
        # Decrease mutation factor as evaluations increase
        return self.base_mutation_factor * (1 - func_evals / self.budget)

    def differential_evolution_step(self, bounds, func, func_evals):
        new_population = np.copy(self.population)
        mutation_factor = self.adaptive_mutation_factor(func_evals)
        for i in range(self.pop_size):
            indices = np.random.choice(range(self.pop_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            mutant_vector = x0 + mutation_factor * (x1 - x2)
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

    def stochastic_local_search(self, bounds, func):
        for i in range(self.pop_size):
            candidate = np.copy(self.population[i])
            perturbation = np.random.normal(0, 0.01, self.dim) * (bounds.ub - bounds.lb)
            candidate += perturbation
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < func(self.population[i]):
                self.population[i] = candidate
                if candidate_fitness < self.best_fitness:
                    self.best_solution = candidate
                    self.best_fitness = candidate_fitness

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        evaluations = 0
        while evaluations < self.budget:
            self.differential_evolution_step(bounds, func, evaluations)
            evaluations += self.pop_size
            if evaluations < self.budget:
                self.stochastic_local_search(bounds, func)
                evaluations += self.pop_size
        return self.best_solution