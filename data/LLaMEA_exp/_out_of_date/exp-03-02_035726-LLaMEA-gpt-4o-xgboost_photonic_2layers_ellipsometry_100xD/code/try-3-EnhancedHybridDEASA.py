import numpy as np

class EnhancedHybridDEASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.velocities = None
        
    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
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
    
    def adaptive_swarm_behavior(self, bounds):
        for i in range(self.pop_size):
            cognitive_term = self.cognitive_coeff * np.random.rand(self.dim) * (self.population[i] - self.population[np.random.randint(self.pop_size)])
            social_term = self.social_coeff * np.random.rand(self.dim) * (self.best_solution - self.population[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] + cognitive_term + social_term)
            self.population[i] = np.clip(self.population[i] + self.velocities[i], bounds.lb, bounds.ub)
    
    def evaluate_population(self, func):
        fitness_values = np.apply_along_axis(func, 1, self.population)
        if np.min(fitness_values) < self.best_fitness:
            self.best_fitness = np.min(fitness_values)
            self.best_solution = self.population[np.argmin(fitness_values)]
        
        return fitness_values
    
    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = 0
        
        while eval_count < self.budget:
            new_population = self.differential_evolution_operator(bounds)
            eval_count += self.pop_size
            
            # Evaluate and select better solutions
            new_fitness_values = np.apply_along_axis(func, 1, new_population)
            eval_count += self.pop_size
            
            for i in range(self.pop_size):
                if new_fitness_values[i] < func(self.population[i]):
                    self.population[i] = new_population[i]
            
            self.adaptive_swarm_behavior(bounds)
            self.evaluate_population(func)
            
        return self.best_solution, self.best_fitness