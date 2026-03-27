import numpy as np

class AHES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        # Initialize population and parameters
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        
        while self.evaluations < self.budget:
            # Dynamic clustering
            clusters = self.dynamic_clustering(population, fitness)
            
            # Local search within clusters
            for cluster in clusters:
                best_individual = cluster[np.argmin([func(ind) for ind in cluster])]
                self.evaluations += len(cluster)
                new_individuals = self.local_search(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                # Update population with improved individuals
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.min(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def dynamic_clustering(self, population, fitness):
        # Simple k-means clustering based on fitness
        num_clusters = min(5, len(population))
        sorted_indices = np.argsort(fitness)
        clusters = np.array_split(population[sorted_indices], num_clusters)
        return clusters
    
    def local_search(self, individual, func):
        # Perturb the individual locally
        perturbations = 0.1 * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        local_population = np.clip(individual + perturbations, func.bounds.lb, func.bounds.ub)
        return local_population