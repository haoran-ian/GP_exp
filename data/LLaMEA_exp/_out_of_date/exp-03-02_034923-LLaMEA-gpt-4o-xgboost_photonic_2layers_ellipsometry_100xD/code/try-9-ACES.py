import numpy as np

class ACES:
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
            # Clustering step to identify promising regions
            num_clusters = max(2, population_size // cluster_factor)
            centroids = self.hierarchical_clustering(population, num_clusters)
            
            # Evolutionary operations
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant_vector = a + mutation_factor * (b - c)
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
    
    def hierarchical_clustering(self, data, k):
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(data, 'ward')
        labels = fcluster(Z, k, criterion='maxclust')
        
        centroids = []
        for i in range(1, k + 1):
            points = data[labels == i]
            if len(points) > 0:
                centroids.append(np.mean(points, axis=0))
        
        return np.array(centroids)