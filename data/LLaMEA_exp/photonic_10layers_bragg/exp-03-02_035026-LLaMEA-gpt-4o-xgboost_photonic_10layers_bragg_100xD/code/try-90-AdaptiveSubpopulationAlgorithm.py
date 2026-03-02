import numpy as np
from sklearn.cluster import KMeans

class AdaptiveSubpopulationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        population_size = 10
        learning_rate = 0.1
        memory = []

        while evaluations < self.budget:
            phase = evaluations / self.budget
            subpopulation_size = self._dynamic_subpopulation_size(phase)

            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.8 * self._dynamic_scaling(phase, memory, best_fitness) / np.sqrt(self.dim), population_size=subpopulation_size, learning_rate=learning_rate)
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_clustering_solutions(best_solution, lb, ub, scale=0.2 * self._fitness_variance(memory) / np.sqrt(self.dim), population_size=subpopulation_size, learning_rate=learning_rate)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_clustering_solutions(best_solution, lb, ub, scale=0.05 * self._fitness_variance(memory) / np.sqrt(self.dim), population_size=subpopulation_size, learning_rate=learning_rate)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_fitness)
            if len(memory) > 20:
                memory.pop(0)

            # Adjust learning rate dynamically based on convergence speed and diversity
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _generate_clustering_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        solutions = np.clip(solutions, lb, ub)
        kmeans = KMeans(n_clusters=min(3, len(solutions)))
        clusters = kmeans.fit_predict(solutions)
        clustered_solutions = np.array([solutions[clusters == i].mean(axis=0) for i in range(kmeans.n_clusters)])
        return clustered_solutions

    def _fitness_variance(self, memory):
        if not memory:
            return 0.1
        return max(0.1, np.std(memory) / 10)

    def _dynamic_scaling(self, phase, memory, best_fitness):
        if not memory:
            return max(0.1, np.abs(best_fitness) / (10 * (1 + phase)))
        return max(0.1, np.mean(memory) / (10 * (1 + phase)))

    def _dynamic_subpopulation_size(self, phase):
        return max(5, int(10 + 10 * (1 - phase)))