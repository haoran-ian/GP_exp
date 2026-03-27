import numpy as np

class DACES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_population_size = 50
        cluster_factor = 5
        mutation_factor = 0.8
        crossover_rate = 0.7
        
        population = np.random.uniform(lb, ub, (initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = initial_population_size
        
        while evaluations < self.budget:
            # Dynamically adjust the population size
            population_size = self.dynamic_population_size(evaluations, initial_population_size)
            
            # Clustering step to identify promising regions
            num_clusters = max(2, population_size // cluster_factor)
            centroids = self.kmeans_clustering(population, num_clusters)
            
            # Evolutionary operations with hybrid mutation strategies
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Hybrid mutation strategy based on cluster centroids
                if i < population_size // 2:
                    mutant_vector = a + mutation_factor * (b - c)
                else:
                    centroid = centroids[np.random.choice(num_clusters)]
                    mutant_vector = a + mutation_factor * (centroid - a)
                
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                crossover = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    mutation_factor = min(1.0, mutation_factor + 0.1)  # Adaptive mutation
                    crossover_rate = max(0.1, crossover_rate - 0.1)    # Adaptive crossover
                else:
                    new_population.append(population[i])
                    mutation_factor = max(0.1, mutation_factor - 0.1)  # Adaptive mutation
                    crossover_rate = min(1.0, crossover_rate + 0.1)    # Adaptive crossover
            
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
    
    def dynamic_population_size(self, evaluations, initial_population_size):
        # Reduce the population size as evaluations increase to emphasize exploitation
        return max(20, int(initial_population_size * (1 - evaluations / self.budget)))