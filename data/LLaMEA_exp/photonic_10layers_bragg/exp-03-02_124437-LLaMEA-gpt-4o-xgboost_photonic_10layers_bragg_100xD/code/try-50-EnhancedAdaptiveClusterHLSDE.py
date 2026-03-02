import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression

class EnhancedAdaptiveClusterHLSDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.base_mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.evaluations = 0
        self.min_clusters = 2
        self.max_clusters = 5
        self.dynamic_factor = 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = self._initialize_population(bounds, self.pop_size)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += len(fitness)

        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while self.evaluations < self.budget:
            # Adaptive clustering
            labels, num_clusters = self._adaptive_clustering(population)
            for cluster in range(num_clusters):
                cluster_indices = np.where(labels == cluster)[0]
                subpop = population[cluster_indices]
                subfitness = fitness[cluster_indices]
                self._differential_evolution(subpop, subfitness, bounds, func, best_solution, cluster)

                if np.min(subfitness) < best_fitness:
                    best_fitness = np.min(subfitness)
                    best_solution = subpop[np.argmin(subfitness)]

                if self.evaluations < self.budget:
                    subpop, subfitness = self._local_model_based_search(subpop, subfitness, bounds, func)

                population[cluster_indices] = subpop
                fitness[cluster_indices] = subfitness

        return best_solution

    def _initialize_population(self, bounds, size):
        return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(size, self.dim)

    def _differential_evolution(self, pop, fitness, bounds, func, best_solution, cluster):
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            adaptive_mutation_factor = self.base_mutation_factor + self.dynamic_factor * (1 - self.evaluations / self.budget)
            mutant = a + adaptive_mutation_factor * (b - c) + 0.1 * np.random.rand() * (best_solution - pop[i])
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

    def _local_model_based_search(self, pop, fitness, bounds, func):
        for i in range(len(pop)):
            model = LinearRegression()
            neighbors = [pop[j] for j in np.random.choice(len(pop), 5, replace=False)]
            X = np.array(neighbors)
            y = np.array([func(x) for x in X])
            self.evaluations += len(neighbors)
            model.fit(X, y)
            direction = model.coef_
            step_size = 0.05 * (1 - (self.evaluations / self.budget))
            candidate = np.clip(pop[i] - step_size * direction, bounds[0], bounds[1])
            candidate_fitness = func(candidate)
            self.evaluations += 1

            if candidate_fitness < fitness[i]:
                pop[i] = candidate
                fitness[i] = candidate_fitness

        return pop, fitness

    def _adaptive_clustering(self, pop):
        if self.evaluations < self.budget / 2:
            num_clusters = self.min_clusters
        else:
            num_clusters = self.max_clusters

        clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(pop)
        return clustering.labels_, num_clusters