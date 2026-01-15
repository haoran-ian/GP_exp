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
        T0 = 1000
        Tf = 1e-2

        # Initialize a population randomly within the bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        # Introduce memory-based archive
        memory = []
        restart_trigger = 0.1

        while self.evaluations < self.budget:
            population_size = max(4, int(10 * self.dim * (1 - self.evaluations / self.budget)))
            learning_rate = 0.1
            diversity = np.mean(np.std(population, axis=0))

            # Adaptive F and CR adjustments
            F = 0.6 + 0.4 * np.sin(np.pi * self.evaluations / self.budget)
            CR = max(0.1, min(1.0, CR + learning_rate * (0.5 - CR) * np.exp(-diversity)))

            # Apply clustering to adaptively select mutation strategies
            num_clusters = int(np.sqrt(population_size))
            kmeans = KMeans(n_clusters=num_clusters).fit(population)
            labels = kmeans.labels_

            for cluster in range(num_clusters):
                cluster_indices = np.where(labels == cluster)[0]
                cluster_size = len(cluster_indices)

                if cluster_size < 3:
                    continue

                for i in cluster_indices:
                    indices = np.random.choice(cluster_indices, 3, replace=False)
                    x1, x2, x3 = population[indices]

                    # Adaptive multi-strategy mutation
                    if np.random.rand() < 0.5:
                        mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                    else:
                        mutant = np.clip(x1 - F * (x2 + x3 - 2 * x1), lb, ub)

                    trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                    trial_fitness = func(trial)
                    self.evaluations += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        memory.append(trial)

                    if self.evaluations >= self.budget:
                        break

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
            
            if diversity < restart_trigger:
                population = np.random.uniform(lb, ub, (population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                self.evaluations += population_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]