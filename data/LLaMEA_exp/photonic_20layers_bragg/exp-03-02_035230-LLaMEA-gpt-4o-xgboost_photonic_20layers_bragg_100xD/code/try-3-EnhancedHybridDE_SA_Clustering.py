import numpy as np
from sklearn.cluster import KMeans

class EnhancedHybridDE_SA_Clustering:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.alpha = 0.9
        self.beta = 0.99
        self.cluster_factor = 3
    
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0
        
        while eval_budget < self.budget:
            # Dynamic clustering to adaptively focus search
            num_clusters = max(2, self.population_size // self.cluster_factor)
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(population)
            cluster_centers = kmeans.cluster_centers_
            
            for center in cluster_centers:
                # Differential Evolution within each cluster
                cluster_indices = np.where(kmeans.labels_ == np.where(kmeans.cluster_centers_ == center)[0][0])[0]
                cluster_pop = population[cluster_indices]
                cluster_fitness = fitness[cluster_indices]
                
                for i in range(len(cluster_pop)):
                    a, b, c = cluster_pop[np.random.choice(len(cluster_pop), 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < self.CR
                    trial = np.where(cross_points, mutant, cluster_pop[i])
                    
                    trial_fitness = func(trial)
                    if eval_budget >= self.budget:
                        break
                    eval_budget += 1
                    if trial_fitness < cluster_fitness[i]:
                        cluster_pop[i] = trial
                        cluster_fitness[i] = trial_fitness
                    else:
                        acceptance_prob = np.exp((cluster_fitness[i] - trial_fitness) / T)
                        if np.random.rand() < acceptance_prob:
                            cluster_pop[i] = trial
                            cluster_fitness[i] = trial_fitness
                
                # Update the cluster population and fitness
                population[cluster_indices] = cluster_pop
                fitness[cluster_indices] = cluster_fitness

            T *= self.alpha

            if np.random.rand() < 0.1:
                self.F = self.F * self.beta + 0.1 * np.random.rand()
                self.CR = self.CR * self.beta + 0.1 * np.random.rand()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]