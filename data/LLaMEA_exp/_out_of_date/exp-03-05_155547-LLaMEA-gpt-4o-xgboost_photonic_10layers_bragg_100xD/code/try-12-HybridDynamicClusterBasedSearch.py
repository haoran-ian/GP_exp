import numpy as np
from sklearn.cluster import KMeans

class HybridDynamicClusterBasedSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Crossover probability
        self.cluster_count = 5  # Number of clusters for dynamic clustering

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.random_state.uniform(lb, ub, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            # Dynamic clustering
            kmeans = KMeans(n_clusters=min(self.cluster_count, len(elite)), random_state=self.random_state)
            labels = kmeans.fit_predict(elite)
            cluster_centers = kmeans.cluster_centers_

            offspring = []
            for _ in range(self.population_size - elite_count):
                cluster_idx = self.random_state.choice(self.cluster_count)
                selected_cluster = elite[labels == cluster_idx]
                if len(selected_cluster) > 1:
                    parent1, parent2 = selected_cluster[self.random_state.choice(len(selected_cluster), 2, replace=False)]
                else:
                    parent1 = parent2 = selected_cluster[0]
                child = self.crossover(parent1, parent2, lb, ub)
                offspring.append(self.mutate(child, lb, ub, evaluations/self.budget, cluster_centers))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub):
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, progress, cluster_centers):
        mutation_strength = self.random_state.rand() * 0.1 * (1 - progress)
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        cluster_center = cluster_centers[self.random_state.choice(len(cluster_centers))]
        mutant = individual + noise + 0.1 * (cluster_center - individual)
        return np.clip(mutant, lb, ub)