import numpy as np
from sklearn.cluster import KMeans

class EDE_PCAPC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        while eval_count < self.budget:
            # Cluster the population to maintain diversity
            num_clusters = max(2, self.population_size // 10)
            kmeans = KMeans(n_clusters=num_clusters).fit(pop)
            labels = kmeans.labels_

            for i in range(self.population_size):
                cluster_indices = np.where(labels == labels[i])[0]
                indices = list(set(cluster_indices) - {i})
                
                if len(indices) < 3: continue  # Skip if not enough individuals in cluster

                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive parameter control based on convergence speed
                if eval_count % (self.population_size * 2) == 0:
                    convergence_speed = (np.max(fitness) - np.min(fitness)) / np.mean(fitness)
                    self.mutation_factor = 0.3 + 0.4 * convergence_speed
                    self.crossover_rate = 0.4 + 0.5 * (1 - convergence_speed)

                if eval_count >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]