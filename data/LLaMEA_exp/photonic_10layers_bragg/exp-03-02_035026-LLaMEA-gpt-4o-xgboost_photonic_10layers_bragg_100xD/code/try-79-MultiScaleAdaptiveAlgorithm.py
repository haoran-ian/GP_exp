import numpy as np

class MultiScaleAdaptiveAlgorithm:
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

            if phase < 0.4:  # Exploration Phase
                scales = [0.8, 0.5, 0.2]
            elif phase < 0.8:  # Balanced Phase
                scales = [0.4, 0.2, 0.1]
            else:  # Exploitation Phase
                scales = [0.1, 0.05, 0.01]

            candidate_solutions = []
            for scale in scales:
                candidate_solutions.extend(
                    self._generate_solutions(best_solution, lb, ub, scale * self._adaptive_scale(phase, memory, best_fitness), population_size // len(scales), learning_rate)
                )

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_fitness)
            if len(memory) > 10:
                memory.pop(0)

            # Adaptive population size and learning rate based on convergence and diversity
            population_size = max(10, int(25 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _adaptive_scale(self, phase, memory, best_fitness):
        if not memory:
            return max(0.05, np.abs(best_fitness) / (10 * (1 + phase)))
        return max(0.05, np.mean(memory) / (10 * (1 + phase)))