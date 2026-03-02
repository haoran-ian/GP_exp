import numpy as np
from sklearn.cluster import KMeans

class ImprovedDynamicClusterDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.evaluations = 0
        self.num_clusters = 5  # Increased number of clusters for more granularity
        self.elite_rate = 0.1  # Percentage of population kept as elite

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = self._initialize_population(bounds, self.pop_size)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(fitness)

        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while self.evaluations < self.budget:
            # Dynamic clustering
            labels = self._dynamic_clustering(population)
            elites = self._retain_elites(population, fitness)
            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                subpop = population[cluster_indices]
                subfitness = fitness[cluster_indices]
                self._differential_evolution(subpop, subfitness, bounds, func, best_solution)

                if np.min(subfitness) < best_fitness:
                    best_fitness = np.min(subfitness)
                    best_solution = subpop[np.argmin(subfitness)]

                if self.evaluations < self.budget:
                    subpop, subfitness = self._enhanced_local_search(subpop, subfitness, bounds, func)

                population[cluster_indices] = subpop
                fitness[cluster_indices] = subfitness

            population, fitness = self._combine_population_elites(population, fitness, elites)

        return best_solution

    def _initialize_population(self, bounds, size):
        return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(size, self.dim)

    def _differential_evolution(self, pop, fitness, bounds, func, best_solution):
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.mutation_factor * (b - c) + 0.1 * np.random.rand() * (best_solution - pop[i])
            mutant = np.clip(mutant, bounds[0], bounds[1])

            cross_points = np.random.rand(self.dim) < self.crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = func(trial)
            self.evaluations += 1

            if trial_fitness < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fitness

    def _enhanced_local_search(self, pop, fitness, bounds, func):
        for i in range(len(pop)):
            perturbation = np.random.normal(0, 0.1, self.dim) * (bounds[1] - bounds[0])
            perturbed = np.clip(pop[i] + perturbation, bounds[0], bounds[1])
            perturbed_fitness = func(perturbed)
            self.evaluations += 1

            if perturbed_fitness < fitness[i]:
                pop[i] = perturbed
                fitness[i] = perturbed_fitness

        return pop, fitness

    def _dynamic_clustering(self, pop):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(pop)
        return kmeans.labels_

    def _retain_elites(self, pop, fitness):
        elite_size = int(self.elite_rate * len(pop))
        elite_indices = np.argsort(fitness)[:elite_size]
        return (pop[elite_indices], fitness[elite_indices])

    def _combine_population_elites(self, pop, fitness, elites):
        elite_pop, elite_fitness = elites
        combined_pop = np.vstack((pop, elite_pop))
        combined_fitness = np.hstack((fitness, elite_fitness))
        sorted_indices = np.argsort(combined_fitness)[:len(pop)]
        return (combined_pop[sorted_indices], combined_fitness[sorted_indices])