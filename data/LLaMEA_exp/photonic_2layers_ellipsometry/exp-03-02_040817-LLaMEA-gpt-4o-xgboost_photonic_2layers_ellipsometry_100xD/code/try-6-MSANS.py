import numpy as np
from sklearn.cluster import KMeans

class MSANS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = []

    def __call__(self, func):
        # Initialize population and parameters
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        while self.evaluations < self.budget:
            # Dynamic clustering with KMeans for neighborhood detection
            num_clusters = min(5, population_size)
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                # Identify the best individual in each cluster
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                # Perform adaptive neighborhood search with memory influence
                new_individuals = self.adaptive_neighborhood_search(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                # Update population using elitism strategy
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
                
                # Update memory with the best individual
                self.memory.append(best_individual)
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def adaptive_neighborhood_search(self, individual, func):
        # Multi-scale adaptive mutation strategy enhanced with memory-based learning
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        global_perturbations = global_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        local_perturbations = local_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        
        # Incorporate memory to refine local search
        if self.memory:
            memory_impact = np.mean(self.memory, axis=0) - individual
            memory_perturbations = 0.1 * memory_impact * np.random.randn(5, self.dim)
            hybrid_perturbations = np.vstack((hybrid_perturbations, memory_perturbations))
        
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population