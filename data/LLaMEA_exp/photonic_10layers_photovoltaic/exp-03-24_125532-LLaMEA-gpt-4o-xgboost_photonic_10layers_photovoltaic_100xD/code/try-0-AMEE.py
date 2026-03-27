import numpy as np

class AMEE:
    def __init__(self, budget, dim):
        self.budget = budget  # Total number of function evaluations allowed
        self.dim = dim  # Dimensionality of the problem
        self.evaluations = 0  # Current number of function evaluations

    def levy_flight(self, size, beta=1.5):
        # Generate Levy flight steps
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def dynamic_cluster(self, population, func):
        # Simple clustering based on similarity in function value
        threshold = np.std([func(ind) for ind in population]) * 0.5
        clusters = []
        cluster_centers = []

        for ind in population:
            if not cluster_centers:
                clusters.append([ind])
                cluster_centers.append(ind)
            else:
                distances = [np.abs(func(ind) - func(center)) for center in cluster_centers]
                if np.min(distances) < threshold:
                    clusters[np.argmin(distances)].append(ind)
                else:
                    clusters.append([ind])
                    cluster_centers.append(ind)

        return clusters

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        # Initialize a random population
        population_size = 50
        population = np.random.uniform(lower_bound, upper_bound, (population_size, self.dim))
        best_solution = None
        best_value = np.inf

        while self.evaluations < self.budget:
            # Evaluate population
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            # Update best solution
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            # Dynamic clustering
            clusters = self.dynamic_cluster(population, func)

            # Exploration and exploitation
            new_population = []
            for cluster in clusters:
                # Exploit best solution in the cluster
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + self.levy_flight(self.dim)
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    new_population.append(new_solution)

                # Explore globally
                if len(new_population) < population_size:
                    new_individual = np.random.uniform(lower_bound, upper_bound, self.dim)
                    new_population.append(new_individual)

            # Update population
            population = np.array(new_population)

        return best_solution, best_value