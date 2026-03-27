import numpy as np

class AdaptiveElitismDynamicPopulation:
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

    def adaptive_cluster(self, population):
        centers = []
        for ind in population:
            if not centers:
                centers.append(ind)
            else:
                scores = [np.linalg.norm(ind - c) for c in centers]
                if min(scores) > np.median(scores) * 0.20:
                    centers.append(ind)
        return centers

    def adaptive_mutation(self, solution, lower_bound, upper_bound, phase_multiplier):
        mutation_strength = np.random.uniform(0.01, 0.15) * phase_multiplier
        mutation_vector = np.random.normal(0, mutation_strength, self.dim)
        mutated_solution = solution + mutation_vector
        return np.clip(mutated_solution, lower_bound, upper_bound)

    def dynamic_population_resize(self, current_population, fitness, target_size):
        elite_indices = np.argsort(fitness)[:target_size]
        return current_population[elite_indices]

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        initial_population_size = 50
        population = np.random.uniform(lower_bound, upper_bound, (initial_population_size, self.dim))
        best_solution = None
        best_value = np.inf

        learning_rate = 0.1
        decay_factor = 0.975
        phase_switch = int(self.budget * 0.3)
        elitism_factor = 0.2  # Starting elitism factor
        resize_frequency = 100  # Frequency of dynamic population resizing

        while self.evaluations < self.budget:
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            centers = self.adaptive_cluster(population)
            clusters = [[ind for ind in population if np.linalg.norm(ind - c) < 0.20] for c in centers]

            new_population = []
            for cluster in clusters:
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + learning_rate * self.levy_flight(self.dim)
                    phase_multiplier = 1.0 if self.evaluations < phase_switch else 1.5
                    new_solution = self.adaptive_mutation(new_solution, lower_bound, upper_bound, phase_multiplier)
                    new_population.append(new_solution)

            elite_size = int(elitism_factor * initial_population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            new_population.extend(elite_individuals)

            if self.evaluations % resize_frequency == 0:
                target_size = max(10, int(elitism_factor * initial_population_size))
                population = self.dynamic_population_resize(population, fitness, target_size)

            if len(new_population) < initial_population_size:
                additional_individuals = np.random.uniform(lower_bound, upper_bound, 
                                                           (initial_population_size - len(new_population), self.dim))
                new_population.extend(additional_individuals)

            population = np.array(new_population)
            if self.evaluations > phase_switch:
                learning_rate *= (decay_factor ** 1.5)
            else:
                learning_rate *= decay_factor

            # Dynamically adjust elitism factor
            elitism_factor = min(0.5, elitism_factor + 0.01)

        return best_solution, best_value