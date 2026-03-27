import numpy as np
from sklearn.cluster import KMeans

class Hybrid_Evolutionary_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize population
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        
        while self.evaluations < self.budget:
            # Dynamic clustering with KMeans
            num_clusters = max(3, int(self.evaluations / self.budget * 15))
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    # Select best and random individuals in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    random_individual = cluster[np.random.randint(cluster.shape[0])]
                    self.evaluations += len(cluster)

                    # Multi-strategy search with hybrid perturbations
                    new_individuals = self.hybrid_perturbation_search(best_individual, random_individual, func)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Update population with elitism
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            inertia_weight = 0.8 - 0.4 * (self.evaluations / self.budget)
                            population[worst_idx] = inertia_weight * population[worst_idx] + (1 - inertia_weight) * ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def hybrid_perturbation_search(self, best_individual, random_individual, func):
        # Adaptive mutation with chaotic maps and differential evolution technique
        exploration_factor = max(0.15, 1.0 - self.evaluations / self.budget)
        exploitation_factor = max(0.1, 0.6 - self.evaluations / (2 * self.budget))
        adaptive_factor = np.sin(np.pi * self.evaluations / self.budget)

        # Chaotic map for perturbation
        chaotic_map = np.random.rand(10, self.dim)
        chaotic_map = 4 * chaotic_map * (1 - chaotic_map)  # Logistic map

        # Global and local perturbations
        global_perturbations = exploration_factor * np.random.randn(5, self.dim) * adaptive_factor * chaotic_map[:5]
        local_perturbations = exploitation_factor * np.random.randn(5, self.dim) * (1 - adaptive_factor) * chaotic_map[5:]
        
        # Differential evolution step
        de_perturbations = exploitation_factor * (best_individual - random_individual) + np.random.randn(5, self.dim)
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations, de_perturbations))
        local_population = np.clip(best_individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population