import numpy as np

class EnhancedAdaptiveOppositionSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        pop_size = min(50, self.budget // 10)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evals = pop_size
        
        while evals < self.budget:
            # Generate reflection population
            reflect_pop = bounds[0] + bounds[1] - population
            reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
            evals += pop_size
            
            # Combine and select the best individuals
            combined_population = np.vstack((population, reflect_pop))
            combined_fitness = np.hstack((fitness, reflect_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            # Adaptive neighborhood-based mutation
            elite = population[:max(1, pop_size // 5)]
            neighborhood_size = 0.05 * (bounds[1] - bounds[0])
            perturbation = np.random.normal(0, neighborhood_size, population.shape)
            new_population = elite + perturbation
            new_population = np.clip(new_population, bounds[0], bounds[1])
            
            # Elitism to preserve best solutions
            population = np.vstack((population, new_population))
            population = population[np.random.choice(population.shape[0], pop_size, replace=False)]
            population[:elite.shape[0]] = elite  # Preserve elite solutions
            
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution