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

    def hierarchical_clustering(self, population, threshold=0.5):
        from scipy.cluster.hierarchy import fclusterdata
        labels = fclusterdata(population, threshold, criterion='distance')
        clusters = [population[labels == i] for i in range(1, max(labels) + 1)]
        return clusters

    def adaptive_mutation(self, solution, lower_bound, upper_bound, phase_multiplier):
        mutation_strength = np.random.uniform(0.01, 0.15) * phase_multiplier
        mutation_vector = np.random.normal(0, mutation_strength, self.dim)
        mutated_solution = solution + mutation_vector
        return np.clip(mutated_solution, lower_bound, upper_bound)

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        initial_population_size = 100  # Increased initial population size
        population = np.random.uniform(lower_bound, upper_bound, (initial_population_size, self.dim))
        best_solution = None
        best_value = np.inf

        learning_rate = 0.1
        decay_factor = 0.95  # Faster decay for more refined search
        phase_switch = int(self.budget * 0.4)  # Phase transition at 40% of budget

        while self.evaluations < self.budget:
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_value:
                best_value = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]

            clusters = self.hierarchical_clustering(population)
            
            new_population = []
            for cluster in clusters:
                if len(cluster) > 0:
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
                new_population.extend(additional_individuals)

            population = np.array(new_population)
            learning_rate *= decay_factor

        return best_solution, best_value