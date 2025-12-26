import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim  # Change: Adjusted population size from 10 * dim to 8 * dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.99
        self.temperature_threshold = 0.1

    def adaptive_cooling(self, current_fitness, best_fitness):
        return max(self.cooling_rate, (best_fitness - current_fitness) / max(best_fitness, 1e-10))

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
        temperature = self.initial_temperature
        
        while evals < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.9 * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < 0.85  # Change: Adjusted crossover probability from 0.9 to 0.85
                offspring = np.where(cross_points, mutant, population[i])
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                evals += 1
                
                # Replacement strategy
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    
                    # Simulated annealing acceptance criterion
                    if offspring_fitness < best_fitness or np.exp((best_fitness - offspring_fitness) / temperature) > np.random.rand():
                        best_solution = offspring
                        best_fitness = offspring_fitness
                
                # Update temperature with adaptive cooling
                temperature *= self.adaptive_cooling(offspring_fitness, best_fitness)
                if temperature < self.temperature_threshold:
                    temperature = self.temperature_threshold

                if evals >= self.budget:
                    break

        return best_solution