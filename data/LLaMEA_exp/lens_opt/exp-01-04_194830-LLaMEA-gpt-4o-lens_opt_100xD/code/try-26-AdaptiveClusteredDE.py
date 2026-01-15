import numpy as np
from sklearn.cluster import KMeans

class AdaptiveClusteredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        F_min, F_max = 0.4, 0.9
        CR_min, CR_max = 0.2, 0.9
        T0 = 1000
        Tf = 1e-2

        # Initialize a population randomly within the bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        # Initialize memory for parameter adaptation
        F_mem = F_max * np.ones(population_size)
        CR_mem = CR_max * np.ones(population_size)

        while self.evaluations < self.budget:
            # Dynamic adjustment of F and CR based on population diversity
            diversity = np.mean(np.std(population, axis=0))
            for i in range(population_size):
                F_mem[i] = F_min + (F_max - F_min) * np.random.rand()
                CR_mem[i] = CR_min + (CR_max - CR_min) * np.random.rand()

            # Apply clustering to adaptively select mutation strategies
            num_clusters = max(2, int(np.sqrt(population_size)))
            kmeans = KMeans(n_clusters=num_clusters).fit(population)
            labels = kmeans.labels_

            for cluster in range(num_clusters):
                cluster_indices = np.where(labels == cluster)[0]
                if len(cluster_indices) < 3:
                    continue

                for i in cluster_indices:
                    indices = np.random.choice(cluster_indices, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    F = F_mem[i]
                    CR = CR_mem[i]
                    
                    mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                    trial_fitness = func(trial)
                    self.evaluations += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness

                    if self.evaluations >= self.budget:
                        break

            # Simulated annealing-like acceptance criterion for diversification
            T = T0 * (Tf / T0) ** (self.evaluations / self.budget)
            for i in range(population_size):
                variance = max(1e-5, diversity)
                neighbor = population[i] + np.random.normal(0, variance, self.dim) * diversity
                neighbor = np.clip(neighbor, lb, ub)
                neighbor_fitness = func(neighbor)
                self.evaluations += 1

                if neighbor_fitness < fitness[i] or np.random.rand() < np.exp(-(neighbor_fitness - fitness[i]) / T):
                    population[i] = neighbor
                    fitness[i] = neighbor_fitness

                if self.evaluations >= self.budget:
                    break
            
            # Reinstate diversity if needed
            if diversity < 1e-5:
                new_population = np.random.uniform(lb, ub, (population_size // 2, self.dim))
                new_fitness = np.array([func(ind) for ind in new_population])
                self.evaluations += len(new_population)
                population = np.vstack((population, new_population))
                fitness = np.hstack((fitness, new_fitness))

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]