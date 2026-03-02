import numpy as np

class EnhancedHybridDEALA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.dynamic_focus_rate = 0.1  # Rate of dynamic focus adjustment
        
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
        
        # Apply dynamic focus adjustment
        dynamic_focus = self.dynamic_focus_rate * (1 - np.arange(self.pop_size) / self.pop_size)
        
        # Focus on promising regions
        for i in range(self.pop_size // 2, self.pop_size):
            self.population[i] = self.population[i] + np.random.normal(0, 0.1 + dynamic_focus[i], self.dim)
            self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)
    
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