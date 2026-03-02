import numpy as np

class ImprovedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.min_population_size = 10
        self.max_population_size = 100
        self.exploration_factor = 0.3
        self.exploitation_factor = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        
        while evaluations < self.budget:
            if np.random.rand() < self.exploration_factor:
                new_population_size = np.random.randint(self.min_population_size, self.max_population_size)
                new_population = np.random.uniform(lb, ub, (new_population_size, self.dim))
            else:
                perturbation = np.random.normal(loc=0.0, scale=np.linspace(0.1, 0.01, population_size)[:, None], size=(population_size, self.dim))
                new_population = population + perturbation * np.random.choice([1, -1], size=(population_size, self.dim))
                new_population = np.clip(new_population, lb, ub)
                
            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += new_population_size

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))

            best_indices = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = population[np.argmin(fitness)]
            
            # Adjust population size based on current evaluations
            population_size = int(self.initial_population_size + (self.max_population_size - self.initial_population_size) * (evaluations / self.budget))
        
        return best_solution