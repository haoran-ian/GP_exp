import numpy as np

class EnhancedMemoryScalingAlgorithm:
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

        scale_factors = {
            'exploration': 0.6,
            'balanced': 0.2,
            'exploitation': 0.05
        }

        while evaluations < self.budget:
            phase = evaluations / self.budget
            scale = self._determine_scale(phase, memory, best_fitness, scale_factors)

            candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size, learning_rate)
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            memory.append(best_fitness)
            if len(memory) > 10:
                memory.pop(0)

            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

    def _determine_scale(self, phase, memory, best_fitness, scale_factors):
        if phase < 0.3:
            scale = scale_factors['exploration'] * self._dynamic_scaling(phase, memory, best_fitness) / np.sqrt(self.dim)
        elif phase < 0.7:
            scale = scale_factors['balanced'] * self._fitness_variance(memory) / np.sqrt(self.dim)
        else:
            scale = scale_factors['exploitation'] * self._fitness_variance(memory) / np.sqrt(self.dim)
        return scale

    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, memory):
        if not memory:
            return 0.1
        return max(0.1, np.std(memory) / 10)

    def _dynamic_scaling(self, phase, memory, best_fitness):
        if not memory:
            return max(0.1, np.abs(best_fitness) / (10 * (1 + phase)))
        return max(0.1, np.mean(memory) / (10 * (1 + phase)))