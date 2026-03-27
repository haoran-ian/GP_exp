import numpy as np

class EnhancedAdaptiveHybridDEALA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.pop_size = self.initial_pop_size
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        
    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        
    def differential_evolution_operator(self, bounds):
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            adaptive_f = self.f * (0.5 + 0.5 * np.cos(np.pi * self.eval_count / self.budget))
            mutant = np.clip(a + adaptive_f * (b - c), bounds.lb, bounds.ub)
            adaptive_cr = self.cr * np.cos(np.pi * self.eval_count / (2 * self.budget))
            cross_points = np.random.rand(self.dim) < adaptive_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            new_population[i] = trial
            
        return new_population
    
    def niche_preservation(self, func):
        niche_radius = 0.1 * (self.budget - self.eval_count) / self.budget
        unique_solutions = []
        for sol in self.population:
            if all(np.linalg.norm(sol - us) > niche_radius for us in unique_solutions):
                unique_solutions.append(sol)
        self.population = np.array(unique_solutions)
        self.pop_size = len(self.population)
    
    def dynamic_population_resizing(self):
        if self.eval_count < self.budget // 2:
            self.pop_size = min(self.initial_pop_size, self.pop_size + 1)
        else:
            self.pop_size = max(self.initial_pop_size // 2, self.pop_size - 1)
    
    def adaptive_landscape_analysis(self, func, bounds):
        fitness_values = np.apply_along_axis(func, 1, self.population)
        self.eval_count += self.pop_size
        
        if np.min(fitness_values) < self.best_fitness:
            self.best_fitness = np.min(fitness_values)
            self.best_solution = self.population[np.argmin(fitness_values)]
            
        sorted_indices = np.argsort(fitness_values)
        self.population = self.population[sorted_indices]
        
        for i in range(self.pop_size // 2, self.pop_size):
            step_size = self.eval_count / self.budget
            self.population[i] = self.population[i] + np.random.normal(0, step_size, self.dim)
            self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)
    
    def hierarchical_local_search(self, solution, func, bounds):
        best_local = solution
        best_local_fitness = func(best_local)
        self.eval_count += 1
        
        for scale in [0.1, 0.01, 0.001]:
            for _ in range(3):
                perturbation = np.random.uniform(-scale, scale, self.dim)
                candidate = np.clip(best_local + perturbation, bounds.lb, bounds.ub)
                candidate_fitness = func(candidate)
                self.eval_count += 1
                
                if candidate_fitness < best_local_fitness:
                    best_local = candidate
                    best_local_fitness = candidate_fitness
        
        return best_local, best_local_fitness

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        while self.eval_count < self.budget:
            self.dynamic_population_resizing()
            new_population = self.differential_evolution_operator(bounds)
            self.eval_count += self.pop_size
            
            new_fitness_values = np.apply_along_axis(func, 1, new_population)
            self.eval_count += self.pop_size
            
            for i in range(self.pop_size):
                if new_fitness_values[i] < func(self.population[i]):
                    self.population[i] = new_population[i]
                if i < self.pop_size // 3:
                    refined_solution, refined_fitness = self.hierarchical_local_search(self.population[i], func, bounds)
                    if refined_fitness < self.best_fitness:
                        self.best_fitness = refined_fitness
                        self.best_solution = refined_solution
            
            self.adaptive_landscape_analysis(func, bounds)
            self.niche_preservation(func)
            
        return self.best_solution, self.best_fitness