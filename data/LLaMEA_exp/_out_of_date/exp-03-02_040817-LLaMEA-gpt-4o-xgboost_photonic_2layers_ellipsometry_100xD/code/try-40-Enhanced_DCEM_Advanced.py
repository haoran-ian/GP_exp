import numpy as np
from sklearn.cluster import KMeans

class Enhanced_DCEM_Advanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.elite_archive = []

    def __call__(self, func):
        # Initialize population and parameters
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        while self.evaluations < self.budget:
            # Dynamic clustering with adaptive number of clusters
            num_clusters = max(2, int(self.evaluations / self.budget * 10))
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    # Identify the best individual in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Store elite individuals in memory
                    self.archive_elite(best_individual, func)

                    # Perform enhanced adaptive search with dynamic mutation
                    new_individuals = self.dynamic_mutation_search(best_individual, func)
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
    
    def archive_elite(self, individual, func):
        # Archive elite based on dynamic competence mapping
        if not self.elite_archive:
            self.elite_archive.append(individual)
        else:
            if func(individual) < func(self.elite_archive[-1]):
                self.elite_archive.append(individual)
                if len(self.elite_archive) > 10:  # Limit archive size
                    self.elite_archive.pop(0)

    def dynamic_mutation_search(self, individual, func):
        # Adaptive mutation strategy for global and local exploration
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        adaptive_factor = self.evaluations / self.budget

        # Use elite archive to guide mutation
        competence_map = np.random.choice(self.elite_archive, size=(10, self.dim)) if self.elite_archive else np.zeros((10, self.dim))

        global_perturbations = global_scale * np.random.randn(5, self.dim) * (1 + adaptive_factor)
        local_perturbations = local_scale * np.random.randn(5, self.dim) * (1 - adaptive_factor)
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations + competence_map[:10], func.bounds.lb, func.bounds.ub)
        return local_population