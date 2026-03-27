import numpy as np
from sklearn.cluster import KMeans

class Enhanced_Swarm_Levy_Dynamic:
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
                    # Best individual in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Multi-strategy search with Lévy flights
                    new_individuals = self.multi_strategy_search(best_individual, func)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Update population with elitism and dynamic learning rates
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            learning_rate = 0.5 + 0.4 * (np.exp(-self.evaluations / self.budget))
                            population[worst_idx] = learning_rate * population[worst_idx] + (1 - learning_rate) * ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_strategy_search(self, individual, func):
        # Adaptive mutation with Lévy flights
        exploration_factor = max(0.1, 1.0 - self.evaluations / self.budget)
        exploitation_factor = max(0.05, 0.5 - self.evaluations / (2 * self.budget))

        # Lévy flight for perturbation
        levy_flights = self.levy_flight(10, self.dim)

        global_perturbations = exploration_factor * np.random.randn(5, self.dim) * levy_flights[:5]
        local_perturbations = exploitation_factor * np.random.randn(5, self.dim) * levy_flights[5:]
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population
    
    def levy_flight(self, num_flies, dim):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        
        u = np.random.randn(num_flies, dim) * sigma
        v = np.random.randn(num_flies, dim)
        step = u / np.abs(v)**(1 / beta)
        
        return step