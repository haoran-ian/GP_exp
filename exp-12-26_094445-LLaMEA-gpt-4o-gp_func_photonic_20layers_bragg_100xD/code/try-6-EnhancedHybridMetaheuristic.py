import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

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
                # Differential Evolution mutation with adaptive parameters
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + np.random.uniform(0.5, 1.0) * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                offspring = np.where(cross_points, mutant, population[i])
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                evals += 1
                
                # Replacement strategy with crowding distance
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    
                    # Simulated annealing acceptance criterion
                    if offspring_fitness < best_fitness or np.exp((best_fitness - offspring_fitness) / self.temperature) > np.random.rand():
                        best_solution = offspring
                        best_fitness = offspring_fitness
                
                # Update temperature
                self.temperature *= self.cooling_rate

                # Early stopping if the budget is exhausted
                if evals >= self.budget:
                    break

            # Introduce crowding distance for diversification
            crowding_distances = np.zeros(self.population_size)
            for j in range(self.dim):
                sorted_idx = np.argsort(population[:, j])
                max_dist = population[sorted_idx[-1], j] - population[sorted_idx[0], j]
                crowding_distances[sorted_idx[0]] = crowding_distances[sorted_idx[-1]] = np.inf
                for k in range(1, self.population_size - 1):
                    crowding_distances[sorted_idx[k]] += (population[sorted_idx[k + 1], j] - population[sorted_idx[k - 1], j]) / max_dist
            
            # Select the most diverse individuals
            diverse_population_idx = np.argsort(crowding_distances)[::-1][:self.population_size]
            population = population[diverse_population_idx]
            fitness = fitness[diverse_population_idx]

        return best_solution