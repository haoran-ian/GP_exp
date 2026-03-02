import numpy as np

class HybridAdaptiveMemoryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        population_size = 15
        learning_rate = 0.2
        memory = []

        while evaluations < self.budget:
            phase = evaluations / self.budget

            if phase < 0.5:  # Hybrid Exploration-Balanced Phase
                scale = 0.6 * self._adaptive_scaling(phase, memory, best_fitness)
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size, learning_rate)
            else:  # Hybrid Balanced-Exploitation Phase
                scale = 0.1 * self._adaptive_scaling(phase, memory, best_fitness)
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size, learning_rate)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_fitness)
            if len(memory) > 20:
                memory.pop(0)

            # Adaptive adjustments for population size and learning rate based on perceived diversity
            population_size = max(5, int(15 - 10 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _adaptive_scaling(self, phase, memory, best_fitness):
        if not memory:
            return max(0.1, np.abs(best_fitness) / (10 * (1 + phase)))
        return max(0.1, np.median(memory) / (10 * (1 + phase)))