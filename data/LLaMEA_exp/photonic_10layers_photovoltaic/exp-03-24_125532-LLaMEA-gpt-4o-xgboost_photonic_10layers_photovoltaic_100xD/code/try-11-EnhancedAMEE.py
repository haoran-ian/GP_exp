import numpy as np
from sklearn.cluster import KMeans

class EnhancedAMEE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def levy_flight(self, size, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def adaptive_fuzzy_cluster(self, population):
        k = max(2, len(population) // 5)
        kmeans = KMeans(n_clusters=k).fit(population)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        clusters = [population[labels == i] for i in range(k)]
        return clusters, centers

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        population_size = 50
        population = np.random.uniform(lower_bound, upper_bound, (population_size, self.dim))
        best_solution = None
        best_value = np.inf

        learning_rate = 0.1
        decay_factor = 0.98

        while self.evaluations < self.budget:
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            clusters, centers = self.adaptive_fuzzy_cluster(population)

            new_population = []
            for cluster, center in zip(clusters, centers):
                new_solutions = []
                if len(cluster) > 1:
                    for _ in range(len(cluster) // 2):
                        new_solution = center + learning_rate * self.levy_flight(self.dim)
                        new_solution = np.clip(new_solution, lower_bound, upper_bound)
                        new_solutions.append(new_solution)
                else:
                    new_solution = center + learning_rate * self.levy_flight(self.dim)
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    new_solutions.append(new_solution)
                new_population.extend(new_solutions)

            elite_size = int(0.1 * population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            new_population.extend(elite_individuals)

            if len(new_population) < population_size:
                additional_individuals = np.random.uniform(lower_bound, upper_bound, (population_size - len(new_population), self.dim))
                new_population.extend(additional_individuals)

            population = np.array(new_population[:population_size])
            learning_rate *= decay_factor

        return best_solution, best_value