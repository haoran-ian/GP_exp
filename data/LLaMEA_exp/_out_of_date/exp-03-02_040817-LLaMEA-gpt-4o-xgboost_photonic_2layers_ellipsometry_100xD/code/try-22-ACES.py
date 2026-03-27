import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class ACES:
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
            # Dynamic clustering with KMeans for adaptive neighborhood detection
            num_clusters = max(2, int(self.evaluations / self.budget * 10))
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            # Adaptive cluster merging based on distance and fitness criteria
            cluster_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
            cluster_fitness = np.array([np.min([func(ind) for ind in cluster]) for cluster in clusters])
            self.evaluations += sum(len(cluster) for cluster in clusters)
            
            distance_matrix = cdist(cluster_centroids, cluster_centroids)
            merged_clusters = []
            merged_indices = set()
            for i in range(num_clusters):
                if i not in merged_indices:
                    close_clusters = np.where((distance_matrix[i] < (func.bounds.ub - func.bounds.lb).mean() * 0.1))[0]
                    best_cluster_idx = close_clusters[np.argmin(cluster_fitness[close_clusters])]
                    if best_cluster_idx != i:
                        merged_clusters.append(np.vstack((clusters[i], clusters[best_cluster_idx])))
                        merged_indices.update(close_clusters)
                    else:
                        merged_clusters.append(clusters[i])
                        merged_indices.add(i)

            for cluster in merged_clusters:
                # Identify the best individual in each cluster
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                # Perform adaptive mutation search with targeted exploitation
                new_individuals = self.targeted_mutation_search(best_individual, func)
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
    
    def targeted_mutation_search(self, individual, func):
        # Enhanced adaptive mutation strategy for global and local exploration
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        adaptive_factor = self.evaluations / self.budget
        global_perturbations = global_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim) * (1 + adaptive_factor)
        local_perturbations = local_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim) * (1 - adaptive_factor)
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population