import numpy as np

class EnhancedACES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        cluster_factor = 5
        mutation_factor = 0.8
        crossover_rate = 0.7
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Clustering step (dynamic number of clusters based on diversity)
            diversity_factor = np.std(population, axis=0).mean()
            num_clusters = max(2, int(population_size // (cluster_factor * diversity_factor + 1)))
            centroids = self.kmeans_clustering(population, num_clusters)
            
            # Random Subspace Sampling
            subspace_dim = max(1, int(0.5 * self.dim))  # Sample 50% of dimensions randomly
            subspace_indices = np.random.choice(self.dim, subspace_dim, replace=False)
            
            # Evolutionary operations
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                # Create mutant vector only in the subspace
                mutant_vector = a.copy()
                mutant_vector[subspace_indices] += mutation_factor * (b[subspace_indices] - c[subspace_indices])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                # Crossover only in the subspace
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
            
            # Elitism: carry over the best solution found so far
            best_idx = np.argmin(fitness)
            new_population[np.random.randint(population_size)] = population[best_idx]
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]
    
    def kmeans_clustering(self, data, k):
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        prev_centroids = centroids.copy()
        
        for _ in range(10):
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