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
        F, CR = 0.8, 0.9
        memory_size = 5
        memory_F, memory_CR = [F] * memory_size, [CR] * memory_size

        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        while self.evaluations < self.budget:
            diversity = np.mean(np.std(population, axis=0))
            idx = self.evaluations % memory_size

            F = np.mean(memory_F) + 0.1 * np.random.standard_normal()
            CR = np.mean(memory_CR) + 0.1 * np.random.standard_normal()

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
                    scaling_factor = 0.5 + 0.3 * np.sin(self.evaluations / self.budget * np.pi * diversity)
                    mutant = np.clip(x1 + scaling_factor * (x2 - x3), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                    trial_fitness = func(trial)
                    self.evaluations += 1

                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        memory_F[idx] = 0.9 * memory_F[idx] + 0.1 * scaling_factor
                        memory_CR[idx] = 0.9 * memory_CR[idx] + 0.1 * CR

                    if self.evaluations >= self.budget:
                        break

            T = 1e-3 + (1 - 1e-3) * (1 - self.evaluations / self.budget)
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