import numpy as np
from sklearn.cluster import KMeans

class Improved_DCEM_Levy_Swarm:
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
            # Adaptive clustering with KMeans for dynamic neighborhood exploration
            num_clusters = max(2, int(self.evaluations / self.budget * 15))
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    # Identify the best individual in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Perform adaptive search with Levy-flight inspired perturbations
                    new_individuals = self.levy_flight_search(best_individual, func)
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
    
    def levy_flight_search(self, individual, func):
        # Levy-flight distribution for more exploratory perturbations
        def levy_distribution(beta=1.5):
            sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                     (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, size=(5, self.dim))
            v = np.random.normal(0, 1, size=(5, self.dim))
            step = u / np.abs(v) ** (1 / beta)
            return step
        
        exploration_factor = max(0.1, 1.0 - self.evaluations / self.budget)
        
        # Generate perturbations using Levy-flight
        perturbations = exploration_factor * levy_distribution()
        local_population = np.clip(individual + perturbations, func.bounds.lb, func.bounds.ub)
        return local_population