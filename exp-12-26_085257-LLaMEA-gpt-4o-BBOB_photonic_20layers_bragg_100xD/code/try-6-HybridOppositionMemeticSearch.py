import numpy as np

class HybridOppositionMemeticSearch:
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
            # Improved opposition-based learning
            reflect_pop = bounds[0] + bounds[1] - population
            reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
            evals += pop_size
            
            # Combine and sort the population by fitness
            combined_population = np.vstack((population, reflect_pop))
            combined_fitness = np.hstack((fitness, reflect_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            # Memetic exploration with adaptive learning rate
            step_size = 0.1 * (bounds[1] - bounds[0]) * (1 - evals / self.budget)
            for i in range(pop_size):
                local_search_candidates = np.random.uniform(bounds[0], bounds[1], (3, self.dim))
                local_search_candidates[0] = population[i]
                local_fitness = np.apply_along_axis(func, 1, local_search_candidates)
                local_best_idx = np.argmin(local_fitness)
                population[i] = local_search_candidates[local_best_idx]
                fitness[i] = local_fitness[local_best_idx]
                
            evals += pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution