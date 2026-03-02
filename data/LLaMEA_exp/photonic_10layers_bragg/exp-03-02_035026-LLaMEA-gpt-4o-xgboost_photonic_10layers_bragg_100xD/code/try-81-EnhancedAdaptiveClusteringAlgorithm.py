import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveClusteringAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        population_size = 15
        learning_rate = 0.1
        memory = []

        while evaluations < self.budget:
            phase = evaluations / self.budget

            if phase < 0.3:  # Exploration Phase with Clustering
                candidate_solutions = self._generate_clustered_solutions(best_solution, lb, ub, scale=0.8, population_size=population_size)
            else:  # Balanced and Exploitation Phases
                scale = 0.2 if phase < 0.7 else 0.05
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=scale * self._fitness_variance(memory), population_size=population_size, learning_rate=learning_rate)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_fitness)
            if len(memory) > 10:
                memory.pop(0)

            # Adjust population size and learning rate dynamically based on convergence speed and diversity
            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _generate_clustered_solutions(self, center, lb, ub, scale, population_size):
        cluster_centers = np.random.uniform(lb, ub, (3, self.dim))
        kmeans = KMeans(n_clusters=3, init=cluster_centers, n_init=1)
        solutions = np.random.uniform(lb, ub, (population_size, self.dim))
        labels = kmeans.fit_predict(solutions)
        scaled_perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        for i in range(population_size):
            solutions[i] = kmeans.cluster_centers_[labels[i]] + scaled_perturbations[i]
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, memory):
        if not memory:
            return 0.1
        return max(0.1, np.std(memory) / 10)