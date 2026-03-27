import numpy as np
from sklearn.cluster import KMeans

class RefinedHybridDEALA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f = 0.8  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        
    def differential_evolution_operator(self, bounds):
        new_population = np.zeros_like(self.population)
        
        for i in range(self.pop_size):
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            new_population[i] = trial
            
        return new_population
    
    def dynamic_landscape_clustering(self, func, bounds):
        fitness_values = np.apply_along_axis(func, 1, self.population)
        if np.min(fitness_values) < self.best_fitness:
            self.best_fitness = np.min(fitness_values)
            self.best_solution = self.population[np.argmin(fitness_values)]
            
        # Cluster solutions using KMeans
        num_clusters = max(2, self.pop_size // 4)
        kmeans = KMeans(n_clusters=num_clusters).fit(self.population)
        cluster_labels = kmeans.labels_
        
        # Focus on promising clusters
        for cluster in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_fitness = fitness_values[cluster_indices]
            best_in_cluster = cluster_indices[np.argmin(cluster_fitness)]
            
            # Perturb population around the best in cluster
            for i in cluster_indices:
                if i != best_in_cluster:
                    self.population[i] = self.population[best_in_cluster] + np.random.normal(0, 0.1, self.dim)
                    self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)
    
    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = 0
        
        while eval_count < self.budget:
            new_population = self.differential_evolution_operator(bounds)
            eval_count += self.pop_size
            
            # Evaluate new_population
            new_fitness_values = np.apply_along_axis(func, 1, new_population)
            eval_count += self.pop_size
            
            # Selection
            for i in range(self.pop_size):
                if new_fitness_values[i] < func(self.population[i]):
                    self.population[i] = new_population[i]
                    
            self.dynamic_landscape_clustering(func, bounds)
            
        return self.best_solution, self.best_fitness