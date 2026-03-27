import numpy as np

class EnhancedHybridDEAL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.fitness_cache = {}

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.fitness_cache.clear()

    def differential_evolution_operator(self, bounds):
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            adaptive_f = self.f * np.exp(-5 * (self.eval_count / self.budget))
            adaptive_cr = self.cr * (1 - np.cos(np.pi * self.eval_count / (2 * self.budget)))
            dynamic_scale = 1 + (self.best_fitness / (self.eval_count + 1))
            mutant = np.clip(a + adaptive_f * (b - c) * dynamic_scale, bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < adaptive_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            new_population[i] = trial
            
        return new_population

    def fitness_inheritance(self, individual, func):
        return self.fitness_cache.get(tuple(individual), func(individual))

    def adaptive_landscape_analysis(self, func, bounds):
        fitness_values = np.array([self.fitness_inheritance(ind, func) for ind in self.population])
        self.eval_count += self.pop_size
        
        if np.min(fitness_values) < self.best_fitness:
            self.best_fitness = np.min(fitness_values)
            self.best_solution = self.population[np.argmin(fitness_values)]
            
        sorted_indices = np.argsort(fitness_values)
        self.population = self.population[sorted_indices]
        self.fitness_cache = {tuple(self.population[i]): fitness_values[i] for i in range(self.pop_size)}
        
        for i in range(self.pop_size // 2, self.pop_size):
            step_size = (self.eval_count / self.budget) ** 0.5
            self.population[i] += np.random.normal(0, step_size, self.dim)
            self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)

    def hierarchical_local_search(self, solution, func, bounds):
        best_local = solution
        best_local_fitness = self.fitness_inheritance(best_local, func)

        for scale in [0.1, 0.01, 0.001]:
            adaptive_scale = scale * (1 - self.eval_count / self.budget)
            for _ in range(3):
                perturbation = np.random.uniform(-adaptive_scale, adaptive_scale, self.dim)
                candidate = np.clip(best_local + perturbation, bounds.lb, bounds.ub)
                candidate_fitness = self.fitness_inheritance(candidate, func)
                self.eval_count += 1
                
                if candidate_fitness < best_local_fitness:
                    best_local = candidate
                    best_local_fitness = candidate_fitness
        
        return best_local, best_local_fitness

    def enhance_exploration_with_multi_scale_perturbation(self, func, bounds):
        for i in range(self.pop_size):
            for scale in [0.04, 0.005]:  # Changed from 0.05 to 0.04
                perturbation = np.random.normal(0, scale, self.dim)
                candidate = np.clip(self.population[i] + perturbation, bounds.lb, bounds.ub)
                candidate_fitness = self.fitness_inheritance(candidate, func)
                self.eval_count += 1
                
                if candidate_fitness < self.fitness_inheritance(self.population[i], func):
                    self.population[i] = candidate
                    self.fitness_cache[tuple(candidate)] = candidate_fitness

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        while self.eval_count < self.budget:
            new_population = self.differential_evolution_operator(bounds)
            new_fitness_values = np.array([self.fitness_inheritance(ind, func) for ind in new_population])
            self.eval_count += self.pop_size
            
            for i in range(self.pop_size):
                if new_fitness_values[i] < self.fitness_inheritance(self.population[i], func):
                    self.population[i] = new_population[i]
                    self.fitness_cache[tuple(self.population[i])] = new_fitness_values[i]
                if i < self.pop_size // 3:
                    refined_solution, refined_fitness = self.hierarchical_local_search(self.population[i], func, bounds)
                    if refined_fitness < self.best_fitness:
                        self.best_fitness = refined_fitness
                        self.best_solution = refined_solution
                        self.fitness_cache[tuple(refined_solution)] = refined_fitness
            
            self.adaptive_landscape_analysis(func, bounds)
            self.enhance_exploration_with_multi_scale_perturbation(func, bounds)
            
        return self.best_solution, self.best_fitness