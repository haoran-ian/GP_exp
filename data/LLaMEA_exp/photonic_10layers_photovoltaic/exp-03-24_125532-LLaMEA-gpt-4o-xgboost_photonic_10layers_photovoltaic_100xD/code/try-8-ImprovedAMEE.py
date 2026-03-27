import numpy as np

class ImprovedAMEE:
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

    def adaptive_cluster(self, population):
        centers = []
        for ind in population:
            if not centers:
                centers.append(ind)
            else:
                # Calculate dissimilarity
                scores = [np.linalg.norm(ind - c) for c in centers]
                if min(scores) > np.mean(scores) * 0.5:
                    centers.append(ind)
        return centers

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        # Initialize a random population
        population_size = 50
        population = np.random.uniform(lower_bound, upper_bound, (population_size, self.dim))
        best_solution = None
        best_value = np.inf

        learning_rate = 0.1  # Initial learning rate
        decay_factor = 0.98  # Slightly faster decay for better adaptation

        while self.evaluations < self.budget:
            # Evaluate population
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            # Update best solution
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            # Adaptive clustering
            centers = self.adaptive_cluster(population)
            clusters = [[ind for ind in population if np.linalg.norm(ind - c) < 0.5] for c in centers]

            # Exploration and exploitation with adaptive learning rate
            new_population = []
            for cluster in clusters:
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + learning_rate * self.levy_flight(self.dim)
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    new_population.append(new_solution)

            # Elitism: retain best individuals
            elite_size = int(0.1 * population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            new_population.extend(elite_individuals)

            # Dynamic population adjustment
            if len(new_population) < population_size:
                additional_individuals = np.random.uniform(lower_bound, upper_bound, (population_size - len(new_population), self.dim))
                new_population.extend(additional_individuals)

            # Update population and learning rate
            population = np.array(new_population)
            learning_rate *= decay_factor  # Decay the learning rate

        return best_solution, best_value