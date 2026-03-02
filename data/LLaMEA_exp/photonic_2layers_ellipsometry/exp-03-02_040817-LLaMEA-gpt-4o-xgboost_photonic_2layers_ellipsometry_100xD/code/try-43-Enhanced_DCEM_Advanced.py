import numpy as np
from sklearn.cluster import KMeans

class Enhanced_DCEM_Advanced:
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
            # Dual-phase adaptive clustering
            exploration_phase = self.evaluations < self.budget * 0.5
            num_clusters = max(2, int((self.evaluations if exploration_phase else self.budget) / self.budget * 10))
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                if cluster.shape[0] > 0:
                    # Identify the best individual in each cluster
                    cluster_fitness = np.array([func(ind) for ind in cluster])
                    best_individual = cluster[np.argmin(cluster_fitness)]
                    self.evaluations += len(cluster)

                    # Enhanced adaptive search with multi-scale stochastic perturbations
                    new_individuals = self.multi_scale_stochastic_search(best_individual, exploration_phase, func)
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    self.evaluations += len(new_individuals)
                    
                    # Diversity-based elitism strategy
                    for i, ind in enumerate(new_individuals):
                        diversity_factor = np.linalg.norm(ind - population, axis=1).mean()
                        if new_fitness[i] < np.max(fitness) and diversity_factor > np.linalg.norm(population[np.argmax(fitness)] - ind):
                            worst_idx = np.argmax(fitness)
                            population[worst_idx] = ind
                            fitness[worst_idx] = new_fitness[i]
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_scale_stochastic_search(self, individual, exploration_phase, func):
        # Enhanced adaptive mutation strategy depending on exploration-exploitation phase
        global_scale = max(0.1, 1.0 - self.evaluations / (2 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / self.budget)
        adaptive_factor = self.evaluations / self.budget

        # Dynamic competence maps for adaptive perturbation control
        competence_map = np.random.choice([0.5, 1.0, 1.5], size=(10, self.dim), p=[0.2, 0.5, 0.3])

        global_perturbations = global_scale * np.random.randn(5, self.dim) * (1 + adaptive_factor) * competence_map[:5]
        local_perturbations = local_scale * np.random.randn(5, self.dim) * (1 - adaptive_factor) * competence_map[5:]
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)
        return local_population