import numpy as np
from sklearn.cluster import KMeans

class Enhanced_Adaptive_Swarm:
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
            # Fitness-proportional dynamic clustering
            fitness_scaled = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-8)
            num_clusters = max(2, int(np.log(1 + np.sum(fitness_scaled)) * 10))
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    # Best individual in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Multi-strategy search with opposition-based learning
                    new_individuals = self.multi_strategy_search(best_individual, func)
                    new_opposite_individuals = self.opposite_population(new_individuals, func)
                    
                    combined_individuals = np.vstack((new_individuals, new_opposite_individuals))
                    combined_fitness = np.array([func(ind) for ind in combined_individuals])
                    self.evaluations += len(combined_individuals)
                    
                    # Update population with elitism and inertia
                    for i, ind in enumerate(combined_individuals):
                        if combined_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            inertia_weight = 0.9 - 0.7 * (self.evaluations / self.budget)
                            population[worst_idx] = inertia_weight * population[worst_idx] + (1 - inertia_weight) * ind
                            fitness[worst_idx] = combined_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_strategy_search(self, individual, func):
        # Adaptive mutation with chaotic maps and inertia
        exploration_factor = max(0.1, 1.0 - 1 / (1 + np.exp(-10 * (self.evaluations / self.budget - 0.5))))
        exploitation_factor = max(0.05, 0.5 - self.evaluations / (2 * self.budget))
        adaptive_factor = np.sin(np.pi * self.evaluations / self.budget)

        # Chaotic map for perturbation
        chaotic_map = np.random.rand(10, self.dim)
        chaotic_map = 4 * chaotic_map * (1 - chaotic_map)  # Logistic map

        global_perturbations = exploration_factor * np.random.randn(5, self.dim) * adaptive_factor * chaotic_map[:5]
        local_perturbations = exploitation_factor * np.random.randn(5, self.dim) * (1 - adaptive_factor) * chaotic_map[5:]
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population
    
    def opposite_population(self, individuals, func):
        # Generate opposite solutions
        opposite_individuals = func.bounds.lb + func.bounds.ub - individuals
        return np.clip(opposite_individuals, func.bounds.lb, func.bounds.ub)