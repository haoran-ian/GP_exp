import numpy as np

class DMCES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 50
        exploration_factor = 0.8
        exploitation_factor = 0.7
        adaptive_rate = 0.1

        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            num_clusters = max(2, int(population_size * (1 - exploration_factor)))
            centroids = self.dynamic_clustering(population, num_clusters)

            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                mutant_vector = a + exploration_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                crossover = np.random.rand(self.dim) < exploitation_factor
                trial_vector = np.where(crossover, mutant_vector, population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    fitness[i] = trial_fitness
                    exploration_factor = min(1.0, exploration_factor + adaptive_rate)
                    exploitation_factor = max(0.1, exploitation_factor - adaptive_rate)
                else:
                    new_population.append(population[i])
                    exploration_factor = max(0.1, exploration_factor - adaptive_rate)
                    exploitation_factor = min(1.0, exploitation_factor + adaptive_rate)

            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def dynamic_clustering(self, data, k):
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        prev_centroids = centroids.copy()

        for _ in range(10):
            distances = np.linalg.norm(data[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            for i in range(k):
                points = data[labels == i]
                if len(points) > 0:
                    centroids[i] = np.mean(points, axis=0)

            if np.all(centroids == prev_centroids):
                break
            prev_centroids = centroids.copy()

        return centroids