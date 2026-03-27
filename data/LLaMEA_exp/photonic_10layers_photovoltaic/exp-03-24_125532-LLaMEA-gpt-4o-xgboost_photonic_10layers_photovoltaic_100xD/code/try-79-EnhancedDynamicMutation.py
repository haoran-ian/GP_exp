import numpy as np

class EnhancedDynamicMutation:
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
                if min(scores) > np.median(scores) * 0.15:  # Adjusted dynamic clustering threshold
                    centers.append(ind)
        return centers

    def hybrid_mutation(self, solution, lower_bound, upper_bound, phase_multiplier):
        gaussian_strength = np.random.uniform(0.01, 0.1) * phase_multiplier
        gaussian_mutation = np.random.normal(0, gaussian_strength, self.dim)
        
        uniform_strength = np.random.uniform(0, 0.05) * phase_multiplier
        uniform_mutation = uniform_strength * (upper_bound - lower_bound)
        
        mutated_solution = solution + gaussian_mutation + uniform_mutation
        return np.clip(mutated_solution, lower_bound, upper_bound)

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        initial_population_size = 60  # Increased population size
        population = np.random.uniform(lower_bound, upper_bound, (initial_population_size, self.dim))
        best_solution = None
        best_value = np.inf

        learning_rate = 0.15  # Modified initial learning rate
        decay_factor = 0.98  # Adjusted decay factor
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
            for cluster in clusters:
                if cluster:
                    cluster_center = np.mean(cluster, axis=0)
                    new_solution = cluster_center + learning_rate * self.levy_flight(self.dim)
                    phase_multiplier = 1.0 if self.evaluations < phase_switch else 2.0  # Adjusted phase multiplier
                    new_solution = self.hybrid_mutation(new_solution, lower_bound, upper_bound, phase_multiplier)
                    new_population.append(new_solution)

            elite_size = int(0.25 * initial_population_size)  # Increased elite size
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            new_population.extend(elite_individuals)

            if len(new_population) < initial_population_size:
                additional_individuals = np.random.uniform(lower_bound, upper_bound, 
                                                           (initial_population_size - len(new_population), self.dim))
                new_population.extend(additional_individuals)

            population = np.array(new_population)
            learning_rate *= decay_factor  # Dynamic learning rate adjustment

        return best_solution, best_value