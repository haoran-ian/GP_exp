import numpy as np

class EnhancedHybridAdaptiveOppositionSearch:
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

        def levy_flight(Lambda):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return step

        while evals < self.budget:
            # Reflective opposition and probabilistic dynamic topology
            reflect_pop = bounds[0] + bounds[1] - population
            rand_pop = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
            new_pop = np.where(np.random.rand(pop_size, self.dim) < 0.5, population + levy_flight(1.5), rand_pop)
            eval_pop = np.vstack((reflect_pop, new_pop))
            eval_fitness = np.apply_along_axis(func, 1, eval_pop)
            evals += eval_pop.shape[0]
            
            # Combine, select and update best
            combined_population = np.vstack((population, reflect_pop, new_pop))
            combined_fitness = np.hstack((fitness, eval_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            # Diversified perturbation mechanism with adaptive step size
            step_size = 0.02 * (bounds[1] - bounds[0])  # Reduced step size for precision
            perturbation = (np.random.randn(*population.shape) * step_size) * np.random.choice([-1, 1], size=population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            # Retain top solutions with diversity enhancement
            population[:2] = combined_population[sorted_indices][:2]
            population[2:] += np.random.normal(0, step_size / 2, population[2:].shape)  # Further diversify
            
            # Evaluate the perturbed population 
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution