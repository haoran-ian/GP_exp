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
        # Enhanced clustering with adaptive cluster sizes
        num_clusters = min(7, len(population))  # Changed from 5 to 7
        sorted_indices = np.argsort(fitness)
        cluster_sizes = [len(population) // num_clusters] * num_clusters  # Calculate cluster sizes
        clusters = [population[sorted_indices[sum(cluster_sizes[:i]):sum(cluster_sizes[:i + 1])]] for i in range(num_clusters)]  # Create clusters
        return clusters
    
    def local_search(self, individual, func):
        # Adaptive mutation based on fitness variability
        mutation_strength = 0.1 * np.std([func(ind) for ind in population])  # Adjust mutation strength
        perturbations = mutation_strength * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)  # Apply mutation
        local_population = np.clip(individual + perturbations, func.bounds.lb, func.bounds.ub)
        return local_population