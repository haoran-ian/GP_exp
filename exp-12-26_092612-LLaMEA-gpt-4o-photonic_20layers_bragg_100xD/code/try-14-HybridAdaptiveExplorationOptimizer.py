import numpy as np
from sklearn.cluster import KMeans

class HybridAdaptiveExplorationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_temp = 100.0
        self.cooling_rate = 0.99  # Increased cooling rate for gradual cooling
        self.bounds = None
        self.elitism_rate = 0.15
        self.cluster_rate = 0.5  # Percentage of population used for clustering
        self.diversity_threshold = 0.1  # Minimum diversity needed for effective exploration

    def _initialize_population(self):
        return np.random.uniform(self.bounds.lb, self.bounds.ub, (self.population_size, self.dim))

    def _calculate_diversity(self, population):
        # Measure population diversity by average pairwise distance
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distances.append(np.linalg.norm(population[i] - population[j]))
        return np.mean(distances)

    def _dynamic_mutation(self, fitness, diversity):
        f_min, f_max = 0.3, 0.8  # Dynamic mutation range
        diversity_factor = max(0, (diversity - self.diversity_threshold) / self.diversity_threshold)
        return f_min + (f_max - f_min) * diversity_factor

    def _mutate(self, target_idx, population, fitness, diversity):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        f = self._dynamic_mutation(fitness[target_idx], diversity)
        mutant = population[a] + f * (population[b] - population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def _crossover(self, target, mutant, cr=0.6):
        mask = np.random.rand(self.dim) < cr
        return np.where(mask, mutant, target)

    def _acceptance_probability(self, current, candidate, t):
        if candidate < current:
            return 1.0
        else:
            return np.exp((current - candidate) / t)

    def _anneal(self, candidate, current, func, temperature):
        candidate_fitness = func(candidate)
        if self._acceptance_probability(func(current), candidate_fitness, temperature) > np.random.rand():
            return candidate, candidate_fitness
        return current, func(current)

    def _cluster_population(self, population, func):
        k = max(2, int(self.cluster_rate * self.population_size))
        kmeans = KMeans(n_clusters=k, n_init=5).fit(population)
        # Return the centroids as representatives of clusters
        cluster_centroids = kmeans.cluster_centers_
        cluster_fitness = np.array([func(ind) for ind in cluster_centroids])
        return cluster_centroids, cluster_fitness

    def __call__(self, func):
        self.bounds = func.bounds
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        temperature = self.initial_temp
        elite_size = int(self.elitism_rate * self.population_size)

        while evaluations < self.budget:
            diversity = self._calculate_diversity(population)
            cluster_centroids, cluster_fitness = self._cluster_population(population, func)
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            for i in range(self.population_size):
                if i < elite_size:
                    # Preserve elite individuals
                    continue
                mutant = self._mutate(i, population, fitness, diversity)
                trial = self._crossover(population[i], mutant)
                new_population[i], new_fitness[i] = self._anneal(trial, population[i], func, temperature)
                evaluations += 1
                if evaluations >= self.budget:
                    break

            # Combine original population, new population, and cluster centroids for selection
            combined_pop = np.vstack((population, new_population, cluster_centroids))
            combined_fitness = np.hstack((fitness, new_fitness, cluster_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_pop[best_indices]
            fitness = combined_fitness[best_indices]

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]