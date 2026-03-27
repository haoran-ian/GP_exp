import numpy as np
from sklearn.cluster import KMeans

class Enhanced_Swarm_Fractal_Chaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize swarm
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        
        while self.evaluations < self.budget:
            # Dynamic clustering with KMeans
            num_clusters = max(2, int(self.evaluations / self.budget * 10))
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Multi-strategy search with fractal and chaos
                    new_individuals = self.fractal_chaos_search(best_individual, func)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Update population with elitism
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            population[worst_idx] = ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def fractal_chaos_search(self, individual, func):
        # Fractal search strategy
        scale_factor = np.random.uniform(low=0.5, high=1.5, size=self.dim)
        fractal_scale = individual * scale_factor
        
        # Chaotic map for diverse search
        chaotic_map = np.random.rand(10, self.dim)
        chaotic_map = 4 * chaotic_map * (1 - chaotic_map)  # Logistic map

        perturbations = fractal_scale * chaotic_map
        local_population = np.clip(individual + perturbations, func.bounds.lb, func.bounds.ub)
        return local_population