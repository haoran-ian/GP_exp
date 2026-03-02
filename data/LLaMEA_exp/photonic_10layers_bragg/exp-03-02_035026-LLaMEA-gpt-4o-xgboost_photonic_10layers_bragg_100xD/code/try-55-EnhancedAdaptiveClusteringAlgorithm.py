import numpy as np

class EnhancedAdaptiveClusteringAlgorithm:
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

            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=self._adaptive_mutation(memory, phase, initial_scale=0.8), population_size=population_size, learning_rate=learning_rate)
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=self._adaptive_mutation(memory, phase, initial_scale=0.2), population_size=population_size, learning_rate=learning_rate)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=self._adaptive_mutation(memory, phase, initial_scale=0.05), population_size=population_size, learning_rate=learning_rate)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_solution)
            if len(memory) > 20:
                memory = self._cluster_memory(memory)

            # Adjust population size and learning rate dynamically based on diversity
            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _adaptive_mutation(self, memory, phase, initial_scale):
        if not memory:
            return initial_scale / (1 + phase)
        return max(0.1, initial_scale * (1 - np.std(memory) / max(1, np.mean(memory))))

    def _cluster_memory(self, memory):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(memory)
        return kmeans.cluster_centers_