import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.population = None
        self.velocities = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.personal_best_positions = None
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        
    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        
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
    
    def particle_swarm_operator(self, bounds):
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social_component = self.c2 * r2 * (self.best_solution - self.population[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
            self.population[i] = np.clip(self.population[i] + self.velocities[i], bounds.lb, bounds.ub)
    
    def adaptive_random_walk(self, bounds):
        for i in range(self.pop_size // 2, self.pop_size):
            step_size = np.random.normal(0, 0.1, self.dim)
            self.population[i] = np.clip(self.population[i] + step_size, bounds.lb, bounds.ub)
    
    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = 0
        
        while eval_count < self.budget:
            new_population = self.differential_evolution_operator(bounds)
            eval_count += self.pop_size
            new_fitness_values = np.apply_along_axis(func, 1, new_population)
            eval_count += self.pop_size
            
            for i in range(self.pop_size):
                if new_fitness_values[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = new_fitness_values[i]
                    self.personal_best_positions[i] = new_population[i]
                if new_fitness_values[i] < self.best_fitness:
                    self.best_fitness = new_fitness_values[i]
                    self.best_solution = new_population[i]
                    
            self.population = np.where(new_fitness_values[:, np.newaxis] < np.apply_along_axis(func, 1, self.population)[:, np.newaxis], new_population, self.population)
            
            self.particle_swarm_operator(bounds)
            self.adaptive_random_walk(bounds)
            
        return self.best_solution, self.best_fitness