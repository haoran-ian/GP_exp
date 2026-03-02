import numpy as np
from sklearn.cluster import KMeans

class Quantum_Adaptive_Swarm:
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

                    # Perform enhanced adaptive search with stochastic jumps and quantum behavior
                    new_individuals = self.quantum_stochastic_search(best_individual, func)
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
    
    def quantum_stochastic_search(self, individual, func):
        # Adaptive mutation strategy using stochastic jumps and quantum behavior
        scale_factor = max(0.1, 1.0 - self.evaluations / self.budget)
        quantum_jumps = scale_factor * np.random.normal(0, 1, (10, self.dim))
        jump_population = np.clip(individual + quantum_jumps, func.bounds.lb, func.bounds.ub)
        
        adaptive_scaling = np.random.uniform(0, scale_factor, (10, self.dim))
        scaled_population = np.clip(individual + adaptive_scaling, func.bounds.lb, func.bounds.ub)
        
        hybrid_population = np.vstack((jump_population, scaled_population))
        return hybrid_population