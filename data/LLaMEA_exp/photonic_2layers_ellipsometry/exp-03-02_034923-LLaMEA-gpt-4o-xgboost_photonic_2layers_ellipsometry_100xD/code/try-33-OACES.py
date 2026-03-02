import numpy as np

class OACES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        cluster_threshold = 0.1
        mutation_factor = 0.8
        crossover_rate = 0.7
        
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        while evaluations < self.budget:
            # Adaptive clustering based on diversity and threshold
            diversity = np.std(population, axis=0).mean()
            num_clusters = max(2, int(population_size * cluster_threshold / (diversity + 1e-9)))
            centroids = self.kmeans_clustering(population, num_clusters)
            
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Hybrid mutation strategy
                if np.random.rand() < 0.5:
                    mutant_vector = a + mutation_factor * (b - c)
                else:
                    mutant_vector = a + mutation_factor * (centroids[np.random.randint(num_clusters)] - a)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                crossover = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    mutation_factor = min(1.0, mutation_factor + 0.05)  # Finer adaptive mutation
                    crossover_rate = max(0.1, crossover_rate - 0.05)    # Finer adaptive crossover
                else:
                    new_population.append(population[i])
                    mutation_factor = max(0.1, mutation_factor - 0.05)  # Finer adaptive mutation
                    crossover_rate = min(1.0, crossover_rate + 0.05)    # Finer adaptive crossover
            
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
            
            if np.allclose(centroids, prev_centroids):
                break
            prev_centroids = centroids.copy()
        
        return centroids