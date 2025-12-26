import numpy as np

class EnhancedHybridAdaptiveOppositionSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        pop_size = min(60, self.budget // 8)  # Increased population size
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evals = pop_size

        def levy_flight(Lambda):
            u = np.random.standard_cauchy(self.dim)  # Cauchy distribution for steps
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return step

        def dynamic_opposition(x):
            return bounds[0] + bounds[1] - x + np.random.uniform(-0.1, 0.1, self.dim)

        while evals < self.budget:
            reflect_pop = dynamic_opposition(population)
            new_pop = population + levy_flight(1.8)  # Tweak Levy exponent
            eval_pop = np.vstack((reflect_pop, new_pop))
            eval_fitness = np.apply_along_axis(func, 1, eval_pop)
            evals += eval_pop.shape[0]
            
            combined_population = np.vstack((population, reflect_pop, new_pop))
            combined_fitness = np.hstack((fitness, eval_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            step_size = 0.05 * (bounds[1] - bounds[0])  # Adjusted step size
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            population[:3] = combined_population[sorted_indices][:3]  # Enhance top solutions retention
            
            fitness = np.apply_along_axis(func, 1, population)
            evals += pop_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
        return best_solution