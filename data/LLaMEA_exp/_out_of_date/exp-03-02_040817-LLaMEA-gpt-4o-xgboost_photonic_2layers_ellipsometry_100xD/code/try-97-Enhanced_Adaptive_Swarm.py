import numpy as np
from sklearn.cluster import KMeans

class Enhanced_Adaptive_Swarm:
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
            fitness_scaled = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-8)
            num_clusters = max(2, int(np.log(1 + np.sum(fitness_scaled)) * 10))
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    new_individuals = self.multi_strategy_search(best_individual, func)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            inertia_weight = 0.9 - 0.7 * (self.evaluations / self.budget)
                            population[worst_idx] = inertia_weight * population[worst_idx] + (1 - inertia_weight) * ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_strategy_search(self, individual, func):
        exploration_factor = max(0.1, 1.0 - 1 / (1 + np.exp(-10 * (self.evaluations / self.budget - 0.5))))
        exploitation_factor = max(0.05, 0.5 - self.evaluations / (2 * self.budget))
        adaptive_factor = np.sin(np.pi * self.evaluations / self.budget)

        chaotic_map = np.random.rand(10, self.dim)
        chaotic_map = 4 * chaotic_map * (1 - chaotic_map)

        global_perturbations = exploration_factor * np.random.randn(5, self.dim) * adaptive_factor * chaotic_map[:5]
        local_perturbations = exploitation_factor * np.random.randn(5, self.dim) * (1 - adaptive_factor) * chaotic_map[5:]
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        
        # Differential Evolution crossover
        for i in range(len(local_population)):
            donor_vector = individual + 0.8 * (global_perturbations[i] - local_perturbations[i])
            crossover_mask = np.random.rand(self.dim) < 0.9
            trial_vector = np.where(crossover_mask, donor_vector, individual)
            local_population[i] = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
        
        return local_population