import numpy as np
from sklearn.cluster import KMeans

class eGLAHS:  # Renamed class for clarity
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = []  # Added memory to store the best solutions

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        self.update_memory(population, fitness)  # Update memory with initial population

        while self.evaluations < self.budget:
            num_clusters = min(5, population_size)
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                new_individuals = self.adaptive_global_local_search(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                self.update_memory(new_individuals, new_fitness)  # Update memory with new individuals
                
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def adaptive_global_local_search(self, individual, func):
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        global_perturbations = global_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        local_perturbations = local_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        resampled_individuals = self.resample_from_memory(func)  # Add strategy to resample from memory
        return np.vstack((local_population, resampled_individuals))

    def update_memory(self, individuals, fitness):
        self.memory.extend(list(zip(individuals, fitness)))
        self.memory.sort(key=lambda x: x[1])
        self.memory = self.memory[:20]  # Keep only top 20 solutions

    def resample_from_memory(self, func):
        if len(self.memory) > 0:
            indices = np.random.choice(len(self.memory), 5, replace=True)
            return np.array([self.memory[i][0] for i in indices])
        else:
            return np.zeros((5, self.dim))