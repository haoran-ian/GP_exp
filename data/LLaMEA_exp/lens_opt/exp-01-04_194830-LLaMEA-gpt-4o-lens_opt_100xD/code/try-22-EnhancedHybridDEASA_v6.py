import numpy as np
from sklearn.cluster import KMeans

class EnhancedHybridDEASA_v6:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        F = 0.8
        CR = 0.9
        adaptive_factor = 0.5  # New adaptive control factor

        # Initialize population randomly within bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size
        memory = []

        while self.evaluations < self.budget:
            # Adaptive adjustment of parameters
            population_size = max(4, int(10 * self.dim * (1 - self.evaluations / self.budget)))
            diversity = np.mean(np.std(population, axis=0))
            learning_rate = 0.1
            F = F + learning_rate * (0.5 - F) * np.exp(-diversity)
            CR = CR + learning_rate * (0.6 - CR) * np.exp(-diversity) * (1 - self.evaluations / self.budget)

            # Clustering for mutation strategy
            num_clusters = int(np.sqrt(population_size))
            kmeans = KMeans(n_clusters=num_clusters).fit(population)
            labels = kmeans.labels_

            for cluster in range(num_clusters):
                cluster_indices = np.where(labels == cluster)[0]
                if len(cluster_indices) < 3:
                    continue

                for i in cluster_indices:
                    indices = np.random.choice(cluster_indices, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    scaling_factor = adaptive_factor + 0.1 * np.sin(self.evaluations / self.budget * np.pi * diversity)
                    mutant = np.clip(x1 + scaling_factor * (x2 - x3), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                    trial_fitness = func(trial)
                    self.evaluations += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        memory.append(trial)

                    if self.evaluations >= self.budget:
                        break

            # Adaptive restart mechanism
            if diversity < 0.1:
                archive_size = min(len(memory), population_size)
                if archive_size > 0:
                    memory_sample = np.array(memory[-archive_size:])
                    population[:archive_size] = memory_sample
                    fitness[:archive_size] = np.array([func(ind) for ind in memory_sample])
                    self.evaluations += archive_size

            # Simulated annealing inspired local search
            T = 1000 * (1e-2 / 1000) ** (self.evaluations / self.budget)
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

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]