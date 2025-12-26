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
            reflect_pop = bounds[0] + bounds[1] - population
            reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
            evals += pop_size
            
            combined_population = np.vstack((population, reflect_pop))
            combined_fitness = np.hstack((fitness, reflect_fitness))
            
            sorted_indices = np.argsort(combined_fitness)
            combined_population = combined_population[sorted_indices]
            combined_fitness = combined_fitness[sorted_indices]
            
            # Selective elitism: keep the best solutions and explore further
            elite_pop = combined_population[:pop_size // 2]
            elite_fitness = combined_fitness[:pop_size // 2]
            
            # Explore the remaining space
            explore_pop = combined_population[pop_size // 2:pop_size]
            explore_fitness = combined_fitness[pop_size // 2:pop_size]
            
            step_size = 0.1 * (bounds[1] - bounds[0]) * (1 - evals/self.budget)
            perturbation = np.random.normal(0, step_size, explore_pop.shape)
            explore_pop += perturbation
            explore_pop = np.clip(explore_pop, bounds[0], bounds[1])
            
            explore_fitness = np.apply_along_axis(func, 1, explore_pop)
            evals += pop_size // 2
            
            # Combine elite and explored populations
            population = np.vstack((elite_pop, explore_pop))
            fitness = np.hstack((elite_fitness, explore_fitness))
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution