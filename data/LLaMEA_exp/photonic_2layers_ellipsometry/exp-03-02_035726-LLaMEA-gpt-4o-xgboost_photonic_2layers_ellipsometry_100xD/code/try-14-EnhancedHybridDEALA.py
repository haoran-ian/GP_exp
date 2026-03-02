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
        
        # Focus on promising regions with increased variance in perturbations
        for i in range(self.pop_size // 2, self.pop_size):
            perturbation_scale = 0.1 if fitness_values[i] > np.median(fitness_values) else 0.05
            self.population[i] = self.population[i] + np.random.normal(0, perturbation_scale, self.dim)
            self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)
    
    def mutation_strategy(self, bounds):
        # Introduce a secondary mutation strategy for exploration
        for i in range(self.pop_size):
            if np.random.rand() < 0.3:  # 30% chance to apply a different mutation
                a, b, c, d = self.population[np.random.choice(self.pop_size, 4, replace=False)]
                mutant = np.clip(a + self.f * (b - c + d - a), bounds.lb, bounds.ub)
                if np.random.rand() < self.cr:
                    self.population[i] = mutant
    
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
            self.mutation_strategy(bounds)
            
        return self.best_solution, self.best_fitness