import numpy as np
from sklearn.cluster import KMeans

class Hybrid_DCEM_Chaotic_Differential_Evolution:
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
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    # Identify the best individual in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Perform hybrid adaptive search with differential evolution and chaotic maps
                    new_individuals = self.hybrid_search(best_individual, func, cluster)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Update population using selective elitism strategy
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            population[worst_idx] = ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def hybrid_search(self, individual, func, cluster):
        # Adaptive mutation strategy using differential evolution and chaotic maps
        exploration_factor = max(0.1, 1.0 - self.evaluations / self.budget)
        exploitation_factor = max(0.05, 0.5 - self.evaluations / (2 * self.budget))
        adaptive_factor = np.sin(np.pi * self.evaluations / self.budget)

        # Differential evolution parameters
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability

        # Chaotic map for perturbation control
        chaotic_map = np.random.rand(10, self.dim)
        chaotic_map = 4 * chaotic_map * (1 - chaotic_map)  # Logistic map

        # Apply differential evolution strategy
        target_idx = np.random.randint(len(cluster))
        other_idxs = np.random.choice([i for i in range(len(cluster)) if i != target_idx], 2, replace=False)
        target_vector = cluster[target_idx]
        donor_vector = target_vector + F * (cluster[other_idxs[0]] - cluster[other_idxs[1]])
        trial_vector = np.where(np.random.rand(self.dim) < CR, donor_vector, target_vector)

        # Apply chaotic perturbations to enhance exploration and exploitation
        global_perturbations = exploration_factor * np.random.randn(5, self.dim) * adaptive_factor * chaotic_map[:5]
        local_perturbations = exploitation_factor * np.random.randn(5, self.dim) * (1 - adaptive_factor) * chaotic_map[5:]

        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        perturbed_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        
        # Combine differential evolution trial vector with perturbed individuals
        combined_population = np.vstack((perturbed_population, trial_vector))
        return combined_population