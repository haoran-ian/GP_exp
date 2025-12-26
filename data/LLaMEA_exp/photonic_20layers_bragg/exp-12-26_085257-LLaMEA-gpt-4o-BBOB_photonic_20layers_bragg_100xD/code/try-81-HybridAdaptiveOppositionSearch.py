import numpy as np

class HybridAdaptiveOppositionSearch:
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
        convergence_rate = np.inf

        def levy_flight(Lambda):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.power(np.abs(v), 1/Lambda)
            return step

        while evals < self.budget:
            # Reflective opposition with Cuckoo Search strategy
            reflect_pop = bounds[0] + bounds[1] - population
            new_pop = population + levy_flight(1.5)
            eval_pop = np.vstack((reflect_pop, new_pop))
            eval_fitness = np.apply_along_axis(func, 1, eval_pop)
            evals += eval_pop.shape[0]
            
            # Combine, select and update best
            combined_population = np.vstack((population, reflect_pop, new_pop))
            combined_fitness = np.hstack((fitness, eval_fitness))
            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices][:initial_pop_size]
            fitness = combined_fitness[sorted_indices][:initial_pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
                new_convergence_rate = (convergence_rate + (fitness[0] - best_fitness)) / 2
            else:
                new_convergence_rate = convergence_rate
            
            # Adaptive mutation scale based on convergence rate
            scaling_factor = max(0.03, min(0.1, convergence_rate / (new_convergence_rate + 1e-9)))
            step_size = scaling_factor * (bounds[1] - bounds[0])
            perturbation = np.random.normal(0, step_size, population.shape)
            population += perturbation
            population = np.clip(population, bounds[0], bounds[1])
            
            # Dynamic population size adjustment
            if evals + population.shape[0] > self.budget:
                population = population[:self.budget - evals]
            
            # Ensure retention of top solutions
            population[:2] = combined_population[sorted_indices][:2]
            
            # Evaluate the perturbed population 
            fitness = np.apply_along_axis(func, 1, population)
            evals += population.shape[0]
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                convergence_rate = new_convergence_rate
                
        return best_solution