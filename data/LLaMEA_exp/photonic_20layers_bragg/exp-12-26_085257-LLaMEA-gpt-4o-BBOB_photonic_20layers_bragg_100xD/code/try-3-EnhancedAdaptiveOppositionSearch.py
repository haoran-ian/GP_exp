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
            reflect_pop = (bounds[0] + bounds[1]) - population + np.random.uniform(-0.1, 0.1, population.shape) * (best_solution - population)
            reflect_fitness = np.apply_along_axis(func, 1, reflect_pop)
            evals += pop_size
            
            combined_pop = np.vstack((population, reflect_pop))
            combined_fitness = np.hstack((fitness, reflect_fitness))
            
            sorted_indices = np.argsort(combined_fitness)
            population = combined_pop[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            elite_size = max(1, pop_size // 5)
            elite_pop = population[:elite_size]
            cov_matrix = np.cov(elite_pop.T) + np.eye(self.dim) * 1e-6
            perturbation = np.random.multivariate_normal(np.zeros(self.dim), cov_matrix, pop_size)
            
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution