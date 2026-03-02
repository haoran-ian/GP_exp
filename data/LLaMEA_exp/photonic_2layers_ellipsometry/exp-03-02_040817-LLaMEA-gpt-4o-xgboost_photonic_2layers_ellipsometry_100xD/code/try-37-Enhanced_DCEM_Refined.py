import numpy as np
from sklearn.cluster import KMeans

class Enhanced_DCEM_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        adaptive_memory = []

        while self.evaluations < self.budget:
            num_clusters = max(2, int(self.evaluations / self.budget * 15))
            kmeans = KMeans(n_clusters=num_clusters)
            labels = kmeans.fit_predict(population)
            clusters = [population[labels == i] for i in range(num_clusters)]
            
            for cluster in clusters:
                cluster_fitness = np.array([func(ind) for ind in cluster])
                best_individual = cluster[np.argmin(cluster_fitness)]
                self.evaluations += len(cluster)

                new_individuals = self.multi_scale_stochastic_search(best_individual, func, adaptive_memory)
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                
                for i, ind in enumerate(new_individuals):
                    if new_fitness[i] < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        population[worst_idx] = ind
                        fitness[worst_idx] = new_fitness[i]

                adaptive_memory.append((best_individual, np.min(cluster_fitness)))
                adaptive_memory = sorted(adaptive_memory, key=lambda x: x[1])[:10]  # Retain top 10 solutions
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]
    
    def multi_scale_stochastic_search(self, individual, func, adaptive_memory):
        global_scale = max(0.1, 1.0 - self.evaluations / (3 * self.budget))
        local_scale = max(0.05, 0.5 - self.evaluations / (2 * self.budget))
        adaptive_factor = self.evaluations / self.budget

        competence_map = np.random.choice([0.5, 1.0, 1.5], size=(10, self.dim), p=[0.2, 0.5, 0.3])

        global_perturbations = global_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim) * (1 + adaptive_factor) * competence_map[:5]
        local_perturbations = local_scale * (func.bounds.ub - func.bounds.lb) * np.random.randn(5, self.dim) * (1 - adaptive_factor) * competence_map[5:]
        
        hybrid_perturbations = np.vstack((global_perturbations, local_perturbations))
        local_population = np.clip(individual + hybrid_perturbations, func.bounds.lb, func.bounds.ub)

        if adaptive_memory:
            memory_based_adjustment = np.mean([mem[0] for mem in adaptive_memory], axis=0)
            local_population += (memory_based_adjustment - individual) * 0.1
        
        return np.clip(local_population, func.bounds.lb, func.bounds.ub)