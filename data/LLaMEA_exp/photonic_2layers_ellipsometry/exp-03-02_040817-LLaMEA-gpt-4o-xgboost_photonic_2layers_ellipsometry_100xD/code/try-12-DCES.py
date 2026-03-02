import numpy as np
from sklearn.cluster import KMeans

class DCES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        while self.evaluations < self.budget:
            num_clusters = max(2, int(self.evaluations / self.budget * 10))
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            # Merge clusters probabilistically for diversity
            if np.random.rand() < 0.3:
                clusters = self.merge_clusters(clusters)
            
            for cluster in clusters:
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                new_individuals = self.dynamic_mutation_search(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def dynamic_mutation_search(self, individual, func):
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        adaptive_factor = self.evaluations / self.budget
        multi_scale = np.array([0.8, 1.0, 1.2])  # Multi-scale factor
        global_perturbations = [
            s * global_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(self.dim) * (1 + adaptive_factor)
            for s in multi_scale
        ]
        local_perturbations = [
            s * local_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(self.dim) * (1 - adaptive_factor)
            for s in multi_scale
        ]
        hybrid_perturbations = np.vstack(global_perturbations + local_perturbations)
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population

    def merge_clusters(self, clusters):
        if len(clusters) < 2:
            return clusters
        idx = np.random.choice(len(clusters), 2, replace=False)
        combined_cluster = np.vstack((clusters[idx[0]], clusters[idx[1]]))
        new_clusters = [clusters[i] for i in range(len(clusters)) if i not in idx]
        new_clusters.append(combined_cluster)
        return new_clusters