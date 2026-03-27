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
        genetic_memory = np.copy(population)
        
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

                    # Multi-strategy search with genetic memory
                    new_individuals = self.multi_strategy_search(best_individual, func, genetic_memory)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Update population with elitism and inertia
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            inertia_weight = 0.9 - 0.7 * (self.evaluations / self.budget)
                            population[worst_idx] = inertia_weight * population[worst_idx] + (1 - inertia_weight) * ind
                            fitness[worst_idx] = new_fitness[i]
                            genetic_memory[worst_idx] = ind  # Update genetic memory with new individual
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_strategy_search(self, individual, func, genetic_memory):
        # Self-adaptive mutation with genetic memory
        mutation_probability = np.random.rand()
        if mutation_probability < 0.1:
            # Use genetic memory for additional diversity
            random_idx = np.random.randint(genetic_memory.shape[0])
            mutation_vector = genetic_memory[random_idx] - individual
        else:
            # Standard adaptive mutation
            mutation_vector = np.random.randn(self.dim)

        exploration_factor = max(0.1, 1.0 - 1 / (1 + np.exp(-10 * (self.evaluations / self.budget - 0.5))))
        exploitation_factor = max(0.05, 0.5 - self.evaluations / (2 * self.budget))
        adaptive_factor = np.sin(np.pi * self.evaluations / self.budget)

        # Chaotic map for perturbation
        chaotic_map = np.random.rand(10, self.dim)
        chaotic_map = 4 * chaotic_map * (1 - chaotic_map)  # Logistic map

        global_perturbations = exploration_factor * mutation_vector * adaptive_factor * chaotic_map[:5]
        local_perturbations = exploitation_factor * mutation_vector * (1 - adaptive_factor) * chaotic_map[5:]
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population