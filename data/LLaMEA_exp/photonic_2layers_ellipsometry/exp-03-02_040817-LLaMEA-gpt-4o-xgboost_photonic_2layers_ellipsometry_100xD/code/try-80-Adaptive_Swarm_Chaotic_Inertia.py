import numpy as np
from sklearn.cluster import KMeans

class Adaptive_Swarm_Chaotic_Inertia:
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

                    # Multi-strategy search
                    new_individuals = self.multi_strategy_search(best_individual, func, population)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Update population with elitism and inertia
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            inertia_weight = 0.9 - 0.7 * (self.evaluations / self.budget)
                            population[worst_idx] = inertia_weight * population[worst_idx] + (1 - inertia_weight) * ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_strategy_search(self, individual, func, population):
        # Adaptive mutation with chaotic maps and inertia
        exploration_factor = max(0.1, 1.0 - 1 / (1 + np.exp(-10 * (self.evaluations / self.budget - 0.5))))
        exploitation_factor = max(0.05, 0.5 - self.evaluations / (2 * self.budget))
        adaptive_factor = np.sin(np.pi * self.evaluations / self.budget)

        # Chaotic map for perturbation
        chaotic_map = np.random.rand(10, self.dim)
        chaotic_map = 4 * chaotic_map * (1 - chaotic_map)

        global_perturbations = exploration_factor * np.random.randn(5, self.dim) * adaptive_factor * chaotic_map[:5]
        local_perturbations = exploitation_factor * np.random.randn(5, self.dim) * (1 - adaptive_factor) * chaotic_map[5:]
        
        # Differential Evolution for enhanced local search
        f = 0.8  # Differential weight
        cr = 0.9  # Crossover probability
        de_perturbations = np.zeros_like(local_perturbations)
        for i in range(local_perturbations.shape[0]):
            idxs = np.random.choice(population.shape[0], 3, replace=False)
            a, b, c = population[idxs]
            mutant = np.clip(a + f * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < cr
            de_perturbations[i] = np.where(cross_points, mutant, local_perturbations[i])

        hybrid_perturbations = np.vstack((global_perturbations, de_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population