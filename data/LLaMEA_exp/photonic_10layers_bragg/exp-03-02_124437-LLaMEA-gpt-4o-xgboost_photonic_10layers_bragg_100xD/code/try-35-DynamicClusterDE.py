import numpy as np
from sklearn.cluster import KMeans

class DynamicClusterDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.evaluations = 0
        self.num_clusters = 3  # Initial number of clusters
        self.adaptive_rate = 0.1  # Introduce adaptive learning rate
        self.stage_threshold = budget // 3  # Divide budget into stages

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = self._initialize_population(bounds, self.pop_size)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(fitness)

        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while self.evaluations < self.budget:
            self._adjust_parameters()  # Adjust parameters based on evaluations
            labels = self._dynamic_clustering(population)
            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                subpop = population[cluster_indices]
                subfitness = fitness[cluster_indices]
                self._differential_evolution(subpop, subfitness, bounds, func, best_solution)

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
            mutant = a + self.mutation_factor * (b - c) + self.adaptive_rate * np.random.rand() * (best_solution - pop[i])
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
        effective_clusters = self.num_clusters + self.evaluations // self.stage_threshold  # Multi-stage clustering
        kmeans = KMeans(n_clusters=effective_clusters, random_state=0).fit(pop)
        return kmeans.labels_

    def _adjust_parameters(self):
        # Adapt mutation factor and clustering based on progress
        if self.evaluations < self.stage_threshold:
            self.mutation_factor = 0.8
        elif self.evaluations < 2 * self.stage_threshold:
            self.mutation_factor = 0.6
        else:
            self.mutation_factor = 0.4