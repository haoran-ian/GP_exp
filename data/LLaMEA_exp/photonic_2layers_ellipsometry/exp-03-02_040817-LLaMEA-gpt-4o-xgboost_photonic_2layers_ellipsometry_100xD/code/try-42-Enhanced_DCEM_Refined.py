import numpy as np
from sklearn.cluster import KMeans

class Enhanced_DCEM_Refined:
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
            num_clusters = max(2, int((1 + np.log(1 + self.evaluations)) / self.budget * 10))  # Adjusted
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    # Identify the best individual in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Perform enhanced adaptive search with multi-scale stochastic perturbations
                    new_individuals = self.multi_scale_stochastic_search(best_individual, func)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Update population using differential elitism strategy
                    for i, ind in enumerate(new_individuals):
                        if new_fitness[i] < np.max(fitness):
                            worst_idx = np.argmax(fitness)
                            population[worst_idx] = ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_scale_stochastic_search(self, individual, func):
        # Enhanced adaptive mutation strategy for global and local exploration
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        adaptive_factor = self.evaluations / self.budget

        # Dynamic competence maps for adaptive perturbation control
        competence_map = np.random.choice([0.5, 1.0, 1.5], size=(10, self.dim), p=[0.2, 0.7, 0.1])  # Adjusted

        global_perturbations = global_scale * np.random.randn(5, self.dim) * (1 + adaptive_factor) * competence_map[:5]
        local_perturbations = local_scale * np.random.randn(5, self.dim) * (1 - adaptive_factor) * competence_map[5:]
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population