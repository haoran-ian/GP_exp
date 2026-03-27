import numpy as np

class AdvancedMultiObjectiveDEAL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.8
        self.cr = 0.9
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.fitness_cache = {}
        self.subcomponents = 3  # Number of subcomponents for cooperative coevolution

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.fitness_cache.clear()

    def dynamic_differential_evolution_operator(self, bounds):
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            # Multi-objective dynamic scaling
            adaptive_f = self.f * (1 + (np.sin(2 * np.pi * self.eval_count / self.budget)))
            adaptive_cr = self.cr * (1 - np.cos(np.pi * self.eval_count / (2 * self.budget)))
            mutant = np.clip(a + adaptive_f * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < adaptive_cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            new_population[i] = trial
            
        return new_population

    def fitness_inheritance(self, individual, func):
        return self.fitness_cache.get(tuple(individual), func(individual))

    def cooperative_coevolution(self, func, bounds):
        subcomponent_size = self.dim // self.subcomponents
        for _ in range(self.subcomponents):
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.subcomponents)]
                np.random.shuffle(indices)
                for sub_idx in indices:
                    sub_start = sub_idx * subcomponent_size
                    sub_end = sub_start + subcomponent_size
                    perturbation = np.random.normal(0, 0.1, subcomponent_size)
                    candidate = np.copy(self.population[i])
                    candidate[sub_start:sub_end] = np.clip(candidate[sub_start:sub_end] + perturbation, 
                                                           bounds.lb[sub_start:sub_end], bounds.ub[sub_start:sub_end])
                    candidate_fitness = self.fitness_inheritance(candidate, func)
                    self.eval_count += 1
                    
                    if candidate_fitness < self.fitness_inheritance(self.population[i], func):
                        self.population[i] = candidate
                        self.fitness_cache[tuple(candidate)] = candidate_fitness

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        
        while self.eval_count < self.budget:
            new_population = self.dynamic_differential_evolution_operator(bounds)
            new_fitness_values = np.array([self.fitness_inheritance(ind, func) for ind in new_population])
            self.eval_count += self.pop_size
            
            for i in range(self.pop_size):
                if new_fitness_values[i] < self.fitness_inheritance(self.population[i], func):
                    self.population[i] = new_population[i]
                    self.fitness_cache[tuple(self.population[i])] = new_fitness_values[i]
                if i < self.pop_size // 3:
                    self.population[i], _ = self.hierarchical_local_search(self.population[i], func, bounds)
            
            self.cooperative_coevolution(func, bounds)
            
            best_idx = np.argmin(new_fitness_values)
            if new_fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = new_fitness_values[best_idx]
                self.best_solution = self.population[best_idx]
        
        return self.best_solution, self.best_fitness