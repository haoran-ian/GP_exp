import numpy as np
from sklearn.cluster import KMeans

class EnhancedDCES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize population and parameters
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        while self.evaluations < self.budget:
            # Two-phase clustering: dynamic and static phases
            phase_ratio = self.evaluations / self.budget
            if phase_ratio < 0.5:
                num_clusters = max(2, int(phase_ratio * 10))
            else:
                num_clusters = 5  # Fixed number for more refined search in later stages
            
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                # Dual mutation strategy: combining exploration and exploitation
                new_individuals = self.dual_mutation_search(best_individual, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                # Update population using enhanced elitism strategy
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def dual_mutation_search(self, individual, func):
        # Enhanced dual mutation strategy for global and local exploration
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        adaptive_factor = self.evaluations / self.budget
        
        # Global mutation for broad exploration
        global_perturbations = global_scale * (func.bounds.ub - func.bounds.lb) * np.random.uniform(-1, 1, (5, self.dim)) * (1 + adaptive_factor)
        
        # Local mutation for fine-grained exploitation
        local_perturbations = local_scale * (func.bounds.ub - func.bounds.lb) * np.random.uniform(-0.5, 0.5, (5, self.dim)) * (1 - adaptive_factor)
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population