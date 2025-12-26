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
            population = combined_population[sorted_indices][:pop_size]
            fitness = combined_fitness[sorted_indices][:pop_size]
            
            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
            
            centroid = np.mean(population, axis=0)
            for i in range(pop_size):
                exploration_vector = np.random.uniform(bounds[0], bounds[1], self.dim)
                new_candidate = 0.5 * (population[i] + centroid) + 0.5 * (exploration_vector - centroid)
                new_candidate = np.clip(new_candidate, bounds[0], bounds[1])
                new_fitness = func(new_candidate)
                evals += 1
                
                if new_fitness < fitness[i]:
                    population[i] = new_candidate
                    fitness[i] = new_fitness

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]
                
            if evals < self.budget // 2:
                pop_size = min(pop_size + 1, self.budget // 5)
            else:
                pop_size = max(10, pop_size - 1)
                
            population = population[:pop_size]
            fitness = fitness[:pop_size]

        return best_solution