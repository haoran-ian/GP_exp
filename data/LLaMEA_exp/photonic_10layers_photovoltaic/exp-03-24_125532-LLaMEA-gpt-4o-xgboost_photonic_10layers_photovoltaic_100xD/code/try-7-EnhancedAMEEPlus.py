import numpy as np

class EnhancedAMEEPlus:
    def __init__(self, budget, dim):
        self.budget = budget  # Total number of function evaluations allowed
        self.dim = dim  # Dimensionality of the problem
        self.evaluations = 0  # Current number of function evaluations

    def levy_flight(self, size, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def dynamic_cluster(self, population, func, adaptive_threshold):
        threshold = np.std([func(ind) for ind in population]) * adaptive_threshold
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

        learning_rate = 0.1  # Initial learning rate
        decay_factor = 0.99  # Decay factor for the learning rate
        adaptive_threshold = 0.5  # Initial adaptive threshold

        while self.evaluations < self.budget:
            # Evaluate population
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            # Update best solution
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            # Dynamic clustering with adaptive threshold
            clusters = self.dynamic_cluster(population, func, adaptive_threshold)
            adaptive_threshold = max(0.1, adaptive_threshold * decay_factor)

            # Exploration and exploitation with adaptive learning rate
            new_population = []
            for cluster in clusters:
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + learning_rate * self.levy_flight(self.dim)
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    new_population.append(new_solution)

            # Multi-level elitism: retain top individuals
            elite_size_primary = int(0.1 * population_size)
            elite_size_secondary = int(0.05 * population_size)
            elite_indices_primary = np.argsort(fitness)[:elite_size_primary]
            elite_indices_secondary = np.argsort(fitness)[elite_size_primary:elite_size_primary + elite_size_secondary]
            elite_individuals_primary = population[elite_indices_primary]
            elite_individuals_secondary = population[elite_indices_secondary]
            new_population.extend(elite_individuals_primary)
            new_population.extend(elite_individuals_secondary)

            # Explore globally if needed
            while len(new_population) < population_size:
                new_individual = np.random.uniform(lower_bound, upper_bound, self.dim)
                new_population.append(new_individual)

            # Update population and learning rate
            population = np.array(new_population)
            learning_rate *= decay_factor  # Decay the learning rate

        return best_solution, best_value