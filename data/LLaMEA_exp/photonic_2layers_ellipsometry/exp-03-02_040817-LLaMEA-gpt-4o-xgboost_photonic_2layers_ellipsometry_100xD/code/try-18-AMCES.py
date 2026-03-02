import numpy as np
from sklearn.cluster import KMeans

class AMCES:
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
            # Multi-cluster dynamic clustering with KMeans for comprehensive neighborhood detection
            num_clusters = max(2, int(self.evaluations / self.budget * 10))
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                # Identify the best individual in each cluster
                if len(cluster) == 0:  # Skip empty clusters
                    continue
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                # Perform adaptive global-local search with progressive neighborhood refinement
                new_individuals = self.progressive_mutation_search(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                # Update population using improved elitism strategy
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def progressive_mutation_search(self, individual, func):
        # Enhanced progressive mutation strategy for thorough exploration and exploitation
        global_scale = max(0.1, 1.0 - self.evaluations / (1.5 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        adaptive_factor = self.evaluations / self.budget
        global_perturbations = global_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(6, self.dim) * (1 + adaptive_factor / 2)
        local_perturbations = local_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(4, self.dim) * (1 - adaptive_factor / 2)
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population