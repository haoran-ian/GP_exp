import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.min_population_size = 5
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.mutation_scale = 0.8

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        # Initialize population
        population_size = self.initial_population_size
        population = np.random.uniform(lower_bound, upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evals = population_size
        
        while evals < self.budget:
            for i in range(population_size):
                # Adaptive mutation scaling
                self.mutation_scale = 0.5 + (0.5 * (self.budget - evals) / self.budget)
                
                # Differential Evolution mutation and crossover
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_scale * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < 0.9
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

            # Dynamic population resizing
            if evals < self.budget and population_size > self.min_population_size:
                top_k = int(population_size * 0.9)
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices[:top_k]]
                fitness = fitness[sorted_indices[:top_k]]
                population_size = len(population)
                
        return best_solution