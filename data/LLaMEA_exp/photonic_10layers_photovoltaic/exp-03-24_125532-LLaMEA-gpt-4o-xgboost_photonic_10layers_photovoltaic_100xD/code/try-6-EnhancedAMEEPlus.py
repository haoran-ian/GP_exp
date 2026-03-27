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

    def dynamic_cluster(self, population, func):
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

    def fitness_sharing(self, fitness, sigma_share=0.5):
        num_individuals = len(fitness)
        shared_fitness = np.copy(fitness)
        for i in range(num_individuals):
            distances = np.abs(fitness - fitness[i])
            sharing_func = np.where(distances < sigma_share, 1 - (distances / sigma_share), 0)
            shared_fitness[i] /= np.sum(sharing_func)
        return shared_fitness

    def adaptive_mutation(self, current_gen, max_gen):
        return 0.1 * (1 - current_gen / max_gen) + 0.01

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        # Initialize a random population
        population_size = 50
        population = np.random.uniform(lower_bound, upper_bound, (population_size, self.dim))
        best_solution = None
        best_value = np.inf

        current_gen = 0
        max_gen = self.budget // population_size

        while self.evaluations < self.budget:
            # Evaluate population
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            # Update best solution
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            # Apply fitness sharing
            shared_fitness = self.fitness_sharing(fitness)

            # Dynamic clustering
            clusters = self.dynamic_cluster(population, func)

            # Exploration and exploitation with adaptive mutation rate
            new_population = []
            mutation_rate = self.adaptive_mutation(current_gen, max_gen)
            for cluster in clusters:
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + mutation_rate * self.levy_flight(self.dim)
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    new_population.append(new_solution)

            # Elitism: retain best individuals
            elite_size = int(0.1 * population_size)
            elite_indices = np.argsort(shared_fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            new_population.extend(elite_individuals)

            # Explore globally if needed
            while len(new_population) < population_size:
                new_individual = np.random.uniform(lower_bound, upper_bound, self.dim)
                new_population.append(new_individual)

            # Update population
            population = np.array(new_population)
            current_gen += 1

        return best_solution, best_value