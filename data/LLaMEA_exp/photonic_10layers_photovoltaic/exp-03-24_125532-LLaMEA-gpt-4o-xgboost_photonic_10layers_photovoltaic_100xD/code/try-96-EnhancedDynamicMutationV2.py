import numpy as np

class EnhancedDynamicMutationV2:
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
                if min(scores) > np.median(scores) * 0.15:  # Further refined dynamic clustering threshold
                    centers.append(ind)
        return centers

    def adaptive_mutation(self, solution, lower_bound, upper_bound, phase_multiplier):
        mutation_strength = np.random.uniform(0.005, 0.10) * phase_multiplier
        mutation_vector = np.random.normal(0, mutation_strength, self.dim)
        mutated_solution = solution + mutation_vector
        return np.clip(mutated_solution, lower_bound, upper_bound)

    def dynamic_learning_rate(self, phase_switch):
        if self.evaluations < phase_switch:
            return 0.1 * ((1 + np.cos(np.pi * self.evaluations / phase_switch)) / 2)  # Cosine annealing for exploration
        else:
            return 0.01 * (0.5 ** ((self.evaluations - phase_switch) / (self.budget - phase_switch)))  # Exponential decay for exploitation
    
    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        initial_population_size = 50
        population = np.random.uniform(lower_bound, upper_bound, (initial_population_size, self.dim))
        best_solution = None
        best_value = np.inf

        decay_factor = 0.985
        phase_switch = int(self.budget * 0.3)

        while self.evaluations < self.budget:
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            centers = self.adaptive_cluster(population)
            clusters = [[ind for ind in population if np.linalg.norm(ind - c) < 0.15] for c in centers]  # Adjusted clustering radius

            new_population = []
            learning_rate = self.dynamic_learning_rate(phase_switch)
            for cluster in clusters:
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + learning_rate * self.levy_flight(self.dim)
                    phase_multiplier = 1.0 if self.evaluations < phase_switch else 1.5
                    new_solution = self.adaptive_mutation(new_solution, lower_bound, upper_bound, phase_multiplier)
                    new_population.append(new_solution)

            elite_size = int(0.2 * initial_population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            new_population.extend(elite_individuals)

            if len(new_population) < initial_population_size:
                additional_individuals = np.random.uniform(lower_bound, upper_bound, 
                                                           (initial_population_size - len(new_population), self.dim))
                additional_individuals *= np.random.uniform(0.9, 1.1, additional_individuals.shape)
                new_population.extend(additional_individuals)

            population = np.array(new_population)

        return best_solution, best_value