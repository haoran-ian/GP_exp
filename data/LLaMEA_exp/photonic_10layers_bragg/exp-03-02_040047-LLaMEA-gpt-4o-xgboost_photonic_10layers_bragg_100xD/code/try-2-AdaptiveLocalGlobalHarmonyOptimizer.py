import numpy as np

class AdaptiveLocalGlobalHarmonyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.harmony_memory_size = 25
        self.exploration_factor = 0.3
        self.exploitation_factor = 0.7
        self.harmony_accept_rate = 0.9
        self.adjust_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        harmony_memory = population[np.argsort(fitness)[:self.harmony_memory_size]]
        
        while evaluations < self.budget:
            new_population = np.empty((self.population_size, self.dim))
            
            for i in range(self.population_size):
                if np.random.rand() < self.harmony_accept_rate:
                    # Harmony search improvisation
                    new_solution = harmony_memory[np.random.randint(self.harmony_memory_size)]
                    if np.random.rand() < self.adjust_rate:
                        new_solution += np.random.normal(0, 0.1, self.dim)
                elif np.random.rand() < self.exploration_factor:
                    new_solution = np.random.uniform(lb, ub, self.dim)
                else:
                    perturbation = np.random.normal(0.0, 0.1, self.dim)
                    new_solution = best_solution + perturbation
                    
                new_solution = np.clip(new_solution, lb, ub)
                new_population[i] = new_solution

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += self.population_size

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))

            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = population[np.argmin(fitness)]

            harmony_memory = population[np.argsort(fitness)[:self.harmony_memory_size]]
        
        return best_solution