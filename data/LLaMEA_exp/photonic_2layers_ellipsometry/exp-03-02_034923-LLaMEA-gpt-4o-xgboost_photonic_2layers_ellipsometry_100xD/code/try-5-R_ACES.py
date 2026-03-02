import numpy as np
from scipy.spatial.distance import cdist

class R_ACES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        cluster_factor = 5
        mutation_factor = 0.8
        crossover_rate = 0.7
        local_search_prob = 0.3
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Clustering step to identify promising regions
            num_clusters = max(2, population_size // cluster_factor)
            centroids = self.kmeans_clustering(population, num_clusters)
            
            # Evolutionary operations with local search
            new_population = []
            for i in range(population_size):
                if np.random.rand() < local_search_prob:
                    # Local search enhancement
                    local_best = self.local_search(population[i], func, lb, ub)
                    new_population.append(local_best)
                    trial_fitness = func(local_best)
                else:
                    # DE-inspired mutation and crossover
                    idxs = [idx for idx in range(population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    
                    mutant_vector = a + mutation_factor * (b - c)
                    mutant_vector = np.clip(mutant_vector, lb, ub)
                    
                    crossover = np.random.rand(self.dim) < crossover_rate
                    trial_vector = np.where(crossover, mutant_vector, population[i])
                    
                    trial_fitness = func(trial_vector)
                    new_population.append(trial_vector)
                
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    mutation_factor = min(1.0, mutation_factor + 0.05)
                    crossover_rate = max(0.1, crossover_rate - 0.05)
                else:
                    mutation_factor = max(0.1, mutation_factor - 0.05)
                    crossover_rate = min(1.0, crossover_rate + 0.05)
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]
    
    def kmeans_clustering(self, data, k):
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        prev_centroids = centroids.copy()
        
        for _ in range(10):
            distances = cdist(data, centroids)
            labels = np.argmin(distances, axis=1)
            
            for i in range(k):
                points = data[labels == i]
                if len(points) > 0:
                    centroids[i] = np.mean(points, axis=0)
            
            if np.allclose(centroids, prev_centroids):
                break
            prev_centroids = centroids.copy()
        
        return centroids
    
    def local_search(self, point, func, lb, ub):
        step_size = 0.01 * (ub - lb)
        local_best = point.copy()
        for _ in range(5):
            perturbation = np.random.uniform(-step_size, step_size)
            candidate = np.clip(local_best + perturbation, lb, ub)
            if func(candidate) < func(local_best):
                local_best = candidate
        return local_best