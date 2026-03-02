import numpy as np
from sklearn.cluster import KMeans

class Refined_DCEM:
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
            # Dynamic clustering with KMeans for adaptive neighborhood detection
            num_clusters = max(2, int(self.evaluations / self.budget * 10))
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                # Identify the best individual in each cluster
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                # Perform refined adaptive search with DE-inspired mutations
                new_individuals = self.de_inspired_mutation(best_individual, cluster, func)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                # Update population using improved elitism strategy
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def de_inspired_mutation(self, best_individual, cluster, func):
        # DE-inspired mutation strategy for global and local exploration
        F = 0.8  # scaling factor for differential weight
        CR = 0.9  # crossover probability

        # Select random individuals for DE mutation
        indices = np.random.choice(len(cluster), 3, replace=False)
        x1, x2, x3 = cluster[indices]
        
        # Generate mutant vector
        mutant_vector = np.clip(x1 + F * (x2 - x3), func.bounds.lb, func.bounds.ub)
        
        # Crossover between the best individual and mutant vector
        crossover_mask = np.random.rand(self.dim) < CR
        trial_vector = np.where(crossover_mask, mutant_vector, best_individual)

        # Apply self-adaptive perturbations
        perturbation = 0.05 * (func.bounds.ub - func.bounds.lb) * np.random.randn(self.dim)
        trial_vector = np.clip(trial_vector + perturbation, func.bounds.lb, func.bounds.ub)
        
        return [trial_vector]