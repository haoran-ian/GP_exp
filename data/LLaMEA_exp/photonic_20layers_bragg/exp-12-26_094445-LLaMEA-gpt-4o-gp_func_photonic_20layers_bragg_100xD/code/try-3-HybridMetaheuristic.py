import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evals = self.population_size
        
        while evals < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.9 * (b - c), lower_bound, upper_bound)  # Adjusted mutation factor
                cross_points = np.random.rand(self.dim) < 0.85  # Adjusted crossover rate
                offspring = np.where(cross_points, mutant, population[i])
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                evals += 1
                
                # Replacement strategy
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    
                    # Simulated annealing acceptance criterion
                    if offspring_fitness < best_fitness or np.exp((best_fitness - offspring_fitness) / self.temperature) > np.random.rand():
                        best_solution = offspring
                        best_fitness = offspring_fitness
                
                # Update temperature
                self.temperature *= self.cooling_rate

                if evals >= self.budget:
                    break

        return best_solution