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
        F_base = 0.8
        CR_base = 0.9

        # Initialize a population randomly within the bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        # Introduce multi-swarm clustering
        num_swarms = min(5, population_size // 10)
        memory = []

        while self.evaluations < self.budget:
            # Dynamic adjustment of population size
            population_size = max(4, int(10 * self.dim * (1 - self.evaluations / self.budget)))
            num_swarms = min(5, population_size // 10)

            # Adaptive control parameters
            diversity = np.mean(np.std(population, axis=0))
            F = F_base * (1 + 0.1 * np.sin(2 * np.pi * self.evaluations / self.budget))
            CR = CR_base * (1 - 0.1 * np.cos(2 * np.pi * self.evaluations / self.budget))

            # Apply clustering to create swarms
            kmeans = KMeans(n_clusters=num_swarms).fit(population)
            labels = kmeans.labels_

            for swarm in range(num_swarms):
                swarm_indices = np.where(labels == swarm)[0]
                if len(swarm_indices) < 3:
                    continue

                for i in swarm_indices:
                    indices = np.random.choice(swarm_indices, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    scaling_factor = F * (0.5 + 0.1 * np.sin(self.evaluations / self.budget * np.pi * diversity))
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

            # Competition-based selection and diversity preservation
            for i in range(population_size):
                variance = max(1e-5, diversity)
                neighbor = population[i] + np.random.normal(0, variance, self.dim)
                neighbor = np.clip(neighbor, lb, ub)
                neighbor_fitness = func(neighbor)
                self.evaluations += 1

                if neighbor_fitness < fitness[i] or np.random.rand() < 0.05:
                    population[i] = neighbor
                    fitness[i] = neighbor_fitness

                if self.evaluations >= self.budget:
                    break

            # Regenerate population if diversity is too low
            if diversity < 0.1:
                population = np.random.uniform(lb, ub, (population_size, self.dim))
                fitness = np.array([func(ind) for ind in population])
                self.evaluations += population_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]