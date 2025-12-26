import numpy as np

class EnhancedHybridAdaptiveOppositionSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_pop_size = min(50, self.budget // 10)
        population = np.random.uniform(bounds[0], bounds[1], (initial_pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evals = initial_pop_size
        
        def levy_flight(Lambda):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return step
        
        def adaptive_population_size(evals, max_evals, init_size):
            return max(5, init_size - int((evals / max_evals) * init_size))
        
        while evals < self.budget:
            current_pop_size = adaptive_population_size(evals, self.budget, initial_pop_size)
            
            reflect_pop = bounds[0] + bounds[1] - population
            levy_exponent = 1.5 + 0.5 * (best_fitness / (best_fitness + np.std(fitness)))
            new_pop = population + levy_flight(levy_exponent)
            eval_pop = np.vstack((reflect_pop, new_pop))
            eval_pop = np.clip(eval_pop, bounds[0], bounds[1])
            eval_fitness = np.apply_along_axis(func, 1, eval_pop)
            evals += eval_pop.shape[0]
            
            combined_population = np.vstack((population, eval_pop))
            combined_fitness = np.hstack((fitness, eval_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:current_pop_size]
            fitness = combined_fitness[sorted_indices][:current_pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            step_size = 0.01 * (bounds[1] - bounds[0])
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            fitness = np.apply_along_axis(func, 1, population)
            evals += current_pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution