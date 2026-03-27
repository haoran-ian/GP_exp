import numpy as np
from sklearn.cluster import KMeans

class E_DC_ES:
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
            # Adaptive clustering based on fitness diversity
            fitness_std = np.std(fitness)
            num_clusters = max(2, min(10, int(fitness_std / np.mean(fitness) * 10)))
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                # Identify the best individual in each cluster
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                # Perform feedback-driven mutation search
                new_individuals = self.feedback_mutation_search(best_individual, func)
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
    
    def feedback_mutation_search(self, individual, func):
        # Feedback-driven adaptive mutation strategy
        success_rate = self.evaluations / self.budget
        global_scale = max(0.05, 1.0 - success_rate)
        local_scale = max(0.02, 0.5 - success_rate)
        global_perturbations = global_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        local_perturbations = local_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim)
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population