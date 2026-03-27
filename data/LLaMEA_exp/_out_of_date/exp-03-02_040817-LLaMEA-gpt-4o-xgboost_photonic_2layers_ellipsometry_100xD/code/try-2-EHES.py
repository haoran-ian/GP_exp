import numpy as np
from sklearn.cluster import KMeans

class EHES:
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
            # Dynamic clustering with KMeans
            num_clusters = min(5, population_size)
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                # Identify the best individual in each cluster
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                # Perform adaptive local search
                new_individuals = self.adaptive_local_search(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                # Update population with improved individuals using elitism
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def adaptive_local_search(self, individual, func):
        # Adaptive mutation strategy
        scale = max(0.1, 1.0 - self.evaluations / self.budget)
        perturbations = scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        local_population = np.clip(individual + perturbations, func.bounds.lb, func.bounds.ub)
        return local_population