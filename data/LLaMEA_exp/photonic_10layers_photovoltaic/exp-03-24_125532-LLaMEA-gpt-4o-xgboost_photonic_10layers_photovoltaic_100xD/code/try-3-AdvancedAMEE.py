import numpy as np

class AdvancedAMEE:
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

    def dynamic_cluster(self, population, func):
        threshold = np.std([func(ind) for ind in population]) * 0.25
        clusters = []
        cluster_centers = []

        for ind in population:
            if not cluster_centers:
                clusters.append([ind])
                cluster_centers.append(ind)
            else:
                distances = [np.linalg.norm(func(ind) - func(center)) for center in cluster_centers]
                if np.min(distances) < threshold:
                    clusters[np.argmin(distances)].append(ind)
                else:
                    clusters.append([ind])
                    cluster_centers.append(ind)

        return clusters

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

            clusters = self.dynamic_cluster(population, func)
            new_population = []
            for cluster in clusters:
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + learning_rate * self.levy_flight(self.dim)
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    new_population.append(new_solution)

            elite_size = int(0.2 * population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            new_population.extend(elite_individuals)

            while len(new_population) < population_size:
                new_individual = np.random.uniform(lower_bound, upper_bound, self.dim)
                new_population.append(new_individual)

            population = np.array(new_population)
            learning_rate *= decay_factor

        return best_solution, best_value