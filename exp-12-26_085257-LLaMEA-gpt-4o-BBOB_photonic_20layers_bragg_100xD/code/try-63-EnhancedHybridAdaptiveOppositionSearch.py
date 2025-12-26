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
        
        while evals < self.budget:
            Lambda = 1.5
            levy_factor = np.random.uniform(0.5, 1.5)
            
            def levy_flight(Lambda, scale):
                u = np.random.normal(0, 1, self.dim)
                v = np.random.normal(0, 1, self.dim)
                step = scale * (u / np.power(np.abs(v), 1/Lambda))
                return step
            
            # Reflective opposition with Cuckoo Search strategy
            reflect_pop = bounds[0] + bounds[1] - population
            new_pop = population + levy_flight(Lambda, levy_factor)
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
            
            # Adaptive step size adjustment for local search
            step_size = 0.02 * (bounds[1] - bounds[0])
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            # Evaluate the perturbed population 
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            # Dynamic population adjustment
            if evals < self.budget * 0.5:
                pop_size = min(pop_size + 1, self.budget // 10)
            elif evals > self.budget * 0.75:
                pop_size = max(pop_size - 1, 10)
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution