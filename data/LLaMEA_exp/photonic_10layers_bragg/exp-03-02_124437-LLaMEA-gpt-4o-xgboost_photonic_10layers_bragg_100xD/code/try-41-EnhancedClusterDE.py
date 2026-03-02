import numpy as np
from sklearn.cluster import KMeans

class EnhancedClusterDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.evaluations = 0
        self.num_clusters = 3  # Initial number of clusters
        self.elite_fraction = 0.1  # Fraction of elites to preserve

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
            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                subpop = population[cluster_indices]
                subfitness = fitness[cluster_indices]

                # Elitist selection
                elites, elite_fitness, non_elites, non_elite_fitness = self._select_elites(subpop, subfitness)

                self._differential_evolution(non_elites, non_elite_fitness, bounds, func, best_solution)
                subpop = np.concatenate((elites, non_elites), axis=0)
                subfitness = np.concatenate((elite_fitness, non_elite_fitness))

                if np.min(subfitness) < best_fitness:
                    best_fitness = np.min(subfitness)
                    best_solution = subpop[np.argmin(subfitness)]

                if self.evaluations < self.budget:
                    subpop, subfitness = self._stochastic_local_search(subpop, subfitness, bounds, func)

                population[cluster_indices] = subpop
                fitness[cluster_indices] = subfitness

        return best_solution

    def _initialize_population(self, bounds, size):
        return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(size, self.dim)

    def _differential_evolution(self, pop, fitness, bounds, func, best_solution):
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            adaptive_mutation = self.mutation_factor * (1 - self.evaluations / self.budget)
            mutant = a + adaptive_mutation * (b - c) + 0.1 * np.random.rand() * (best_solution - pop[i])
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

    def _stochastic_local_search(self, pop, fitness, bounds, func):
        for i in range(len(pop)):
            perturbation = np.random.normal(0, 0.05, self.dim) * (bounds[1] - bounds[0])
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

    def _select_elites(self, pop, fitness):
        elite_count = int(len(pop) * self.elite_fraction)
        elite_indices = np.argsort(fitness)[:elite_count]
        non_elite_indices = np.argsort(fitness)[elite_count:]

        elites = pop[elite_indices]
        elite_fitness = fitness[elite_indices]
        non_elites = pop[non_elite_indices]
        non_elite_fitness = fitness[non_elite_indices]

        return elites, elite_fitness, non_elites, non_elite_fitness