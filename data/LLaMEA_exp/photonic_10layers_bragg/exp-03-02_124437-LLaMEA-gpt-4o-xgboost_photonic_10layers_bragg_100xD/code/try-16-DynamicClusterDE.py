import numpy as np
from sklearn.cluster import KMeans

class DynamicClusterDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.mutation_factor = 0.5 + np.random.rand() * 0.5  # Adaptive mutation factor
        self.crossover_prob = 0.7 + np.random.rand() * 0.3  # Adaptive crossover probability
        self.evaluations = 0
        self.alpha = 0.2  # Exploration factor adjustment

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = self._initialize_population(bounds, self.pop_size)

        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, population)
            self.evaluations += len(fitness)

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx]

            num_clusters = max(2, self.pop_size // 10)
            population, fitness = self._dynamic_clustering(population, fitness, bounds, func, num_clusters)

            if self.evaluations < self.budget:
                population, fitness = self._adaptive_local_search(population, fitness, bounds, func)

        return best_solution

    def _initialize_population(self, bounds, size):
        return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(size, self.dim)

    def _dynamic_clustering(self, pop, fitness, bounds, func, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        labels = kmeans.fit_predict(pop)
        new_pop = np.copy(pop)

        for cluster in range(num_clusters):
            cluster_indices = np.where(labels == cluster)[0]
            if len(cluster_indices) < 3:
                continue
            
            for i in cluster_indices:
                idxs = [idx for idx in cluster_indices if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c) + self.alpha * np.random.rand() * (kmeans.cluster_centers_[cluster] - pop[i])
                mutant = np.clip(mutant, bounds[0], bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness

        return new_pop, fitness

    def _adaptive_local_search(self, pop, fitness, bounds, func):
        for i in range(len(pop)):
            perturbation = np.random.normal(0, 0.05, self.dim) * (bounds[1] - bounds[0])
            perturbed = np.clip(pop[i] + perturbation, bounds[0], bounds[1])
            perturbed_fitness = func(perturbed)
            self.evaluations += 1
            
            if perturbed_fitness < fitness[i]:
                pop[i] = perturbed
                fitness[i] = perturbed_fitness
        
        return pop, fitness