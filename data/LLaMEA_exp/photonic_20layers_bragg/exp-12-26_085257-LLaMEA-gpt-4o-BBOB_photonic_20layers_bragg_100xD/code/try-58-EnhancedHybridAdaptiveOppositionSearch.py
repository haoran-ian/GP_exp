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

        def dynamic_levy_flight(iteration, max_iters):
            Lambda = 1.5 + 0.5 * (1 - iteration / max_iters)  # Dynamic adjustment
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return step

        iteration = 0
        max_iters = self.budget // pop_size

        while evals < self.budget:
            iteration += 1
            # Reflective opposition with Cuckoo Search strategy
            reflect_pop = bounds[0] + bounds[1] - population
            new_pop = population + dynamic_levy_flight(iteration, max_iters)
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
            
            # Diversity preservation through mutation
            diversity_factor = 0.1 * (bounds[1] - bounds[0])
            mutation = np.random.normal(0, diversity_factor, population.shape)
            population += mutation
            population = np.clip(population, bounds[0], bounds[1])
            
            # Evaluate the mutated population 
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution