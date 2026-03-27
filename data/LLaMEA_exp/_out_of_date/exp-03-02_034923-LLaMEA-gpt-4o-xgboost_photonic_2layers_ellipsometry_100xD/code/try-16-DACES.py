import numpy as np

class DACES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        min_cluster_factor = 2
        max_cluster_factor = 10
        mutation_factor = np.full(population_size, 0.8)
        crossover_rate = np.full(population_size, 0.7)
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Dynamic clustering factor based on diversity
            cluster_factor = np.clip(max_cluster_factor - int(np.std(fitness) * 10), min_cluster_factor, max_cluster_factor)
            num_clusters = max(2, population_size // cluster_factor)
            centroids = self.kmeans_clustering(population, num_clusters)
            
            # Evolutionary operations
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant_vector = a + mutation_factor[i] * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                crossover = np.random.rand(self.dim) < crossover_rate[i]
                trial_vector = np.where(crossover, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    mutation_factor[i] = min(1.0, mutation_factor[i] + 0.05)  # Adaptive mutation
                    crossover_rate[i] = max(0.1, crossover_rate[i] - 0.05)   # Adaptive crossover
                else:
                    new_population.append(population[i])
                    mutation_factor[i] = max(0.1, mutation_factor[i] - 0.05)  # Adaptive mutation
                    crossover_rate[i] = min(1.0, crossover_rate[i] + 0.05)   # Adaptive crossover
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]
    
    def kmeans_clustering(self, data, k):
        # Simple k-means clustering to get cluster centroids
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        prev_centroids = centroids.copy()
        
        for _ in range(10):  # Run for a fixed number of iterations
            distances = np.linalg.norm(data[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            for i in range(k):
                points = data[labels == i]
                if len(points) > 0:
                    centroids[i] = np.mean(points, axis=0)
            
            if np.all(centroids == prev_centroids):
                break
            prev_centroids = centroids.copy()
        
        return centroids