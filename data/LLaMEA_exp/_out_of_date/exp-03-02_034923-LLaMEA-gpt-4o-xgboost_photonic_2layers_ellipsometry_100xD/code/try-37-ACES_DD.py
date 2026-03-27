import numpy as np

class ACES_DD:
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
            # Calculate diversity metrics and adjust strategies
            diversity = self.calculate_diversity(population)
            self.adjust_strategies(diversity)
            
            # Dynamic clustering step
            num_clusters = max(2, int(population_size // (cluster_factor * diversity.mean() + 1)))
            centroids = self.kmeans_clustering(population, num_clusters)
            
            # Evolutionary operations with adaptive features
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                
                mutant_vector = a + mutation_factor * (b - c) * diversity
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                crossover = np.random.rand(self.dim) < (crossover_rate * diversity)
                trial_vector = np.where(crossover, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])
            
            # Elitism: carry over the best solution found so far
            best_idx = np.argmin(fitness)
            new_population[np.random.randint(population_size)] = population[best_idx]
            
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

    def calculate_diversity(self, population):
        # Calculate diversity metric as standard deviation across dimensions
        return np.std(population, axis=0)
    
    def adjust_strategies(self, diversity):
        # Adjust mutation and crossover rates based on diversity
        global mutation_factor, crossover_rate
        if diversity.mean() > 0.5:
            mutation_factor = max(0.5, mutation_factor + 0.1)
            crossover_rate = min(0.9, crossover_rate - 0.1)
        else:
            mutation_factor = min(1.0, mutation_factor - 0.1)
            crossover_rate = max(0.1, crossover_rate + 0.1)