import numpy as np

class RefinedHybridDEALA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.elite_size = 5  # Elite retention size
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        
    def differential_evolution_operator(self, bounds):
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            new_population[i] = trial
            
        return new_population
    
    def adaptive_landscape_analysis(self, func, bounds):
        fitness_values = np.apply_along_axis(func, 1, self.population)
        if np.min(fitness_values) < self.best_fitness:
            self.best_fitness = np.min(fitness_values)
            self.best_solution = self.population[np.argmin(fitness_values)]
            
        # Sort solutions by fitness
        sorted_indices = np.argsort(fitness_values)
        self.population = self.population[sorted_indices]
        
        # Retain elite solutions
        elite = self.population[:self.elite_size].copy()
        
        # Focus on promising regions dynamically
        for i in range(self.elite_size, self.pop_size):
            perturbation = np.random.normal(0, 0.1 * (self.pop_size - i) / self.pop_size, self.dim)
            self.population[i] = self.population[i] + perturbation
            self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)
        
        # Reinstate elite solutions
        self.population[:self.elite_size] = elite

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = 0
        
        while eval_count < self.budget:
            new_population = self.differential_evolution_operator(bounds)
            eval_count += self.pop_size
            
            # Evaluate new_population
            new_fitness_values = np.apply_along_axis(func, 1, new_population)
            eval_count += self.pop_size
            
            # Selection
            for i in range(self.pop_size):
                if new_fitness_values[i] < func(self.population[i]):
                    self.population[i] = new_population[i]

            self.adaptive_landscape_analysis(func, bounds)
            
        return self.best_solution, self.best_fitness