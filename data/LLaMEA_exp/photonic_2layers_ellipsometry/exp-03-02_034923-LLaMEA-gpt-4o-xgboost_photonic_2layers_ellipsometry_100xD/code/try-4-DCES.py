import numpy as np
from scipy.spatial import distance

class DCES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        mutation_factor = 0.8
        crossover_rate = 0.7

        # Initial population
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            # Dynamic clustering based on fitness to identify promising regions
            num_clusters = self.dynamic_clustering(population, fitness)
            centroids = self.kmeans_clustering(population, num_clusters)
            
            # Evolutionary operations with diversity maintenance
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
                else:
                    new_population.append(population[i])
                    mutation_factor = max(0.1, mutation_factor - 0.1)  # Adaptive mutation
            
            if len(new_population) < population_size:
                extra = np.random.uniform(lb, ub, (population_size - len(new_population), self.dim))
                new_population.extend(extra)
                
            # Adaptive crossover based on diversity
            diversity = self.calculate_diversity(new_population)
            crossover_rate = 0.5 + 0.5 * (diversity / self.dim)
            
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]

    def kmeans_clustering(self, data, k):
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        prev_centroids = centroids.copy()
        
        for _ in range(10):  # Fixed number of iterations for clustering
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

    def dynamic_clustering(self, population, fitness):
        # Determine number of clusters dynamically based on fitness distribution
        fitness_std = np.std(fitness)
        num_clusters = max(2, int(len(population) * (1 + fitness_std / np.max(fitness_std))))
        return num_clusters

    def calculate_diversity(self, population):
        # Calculate average pairwise distance as a measure of diversity
        return np.mean(distance.pdist(population))