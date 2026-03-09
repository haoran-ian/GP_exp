import numpy as np
from sklearn.cluster import KMeans

class EnhancedDynamicPopulationStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.min_cr = 0.6  # Minimum crossover probability
        self.max_cr = 0.9  # Maximum crossover probability
        self.min_mutation_strength = 0.05
        self.max_mutation_strength = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = self.random_state.uniform(lb, ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            # Cluster the current population to assess diversity
            if len(population) > 1:
                num_clusters = max(2, population_size // 10)
                kmeans = KMeans(n_clusters=num_clusters, random_state=self.random_state)
                kmeans.fit(population)
                cluster_centers = kmeans.cluster_centers_
                diversity = np.mean(np.linalg.norm(cluster_centers - np.mean(cluster_centers, axis=0), axis=1))
            else:
                diversity = 0

            offspring = []
            for _ in range(population_size - elite_count):
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub, diversity)
                offspring.append(self.mutate(child, lb, ub, diversity))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Dynamically adjust population size
            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub, diversity):
        cr = self.min_cr + (self.max_cr - self.min_cr) * (1 - diversity)
        mask = self.random_state.rand(self.dim) < cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, diversity):
        mutation_strength = self.min_mutation_strength + (self.max_mutation_strength - self.min_mutation_strength) * (1 - diversity)
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)