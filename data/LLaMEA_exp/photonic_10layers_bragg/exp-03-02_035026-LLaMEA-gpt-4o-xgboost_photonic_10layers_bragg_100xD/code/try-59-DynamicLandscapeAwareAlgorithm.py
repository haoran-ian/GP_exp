import numpy as np

class DynamicLandscapeAwareAlgorithm:
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

            # Dynamically adjust exploration based on local landscape features
            local_exp_factor = self._local_exploration_factor(func, best_solution, lb, ub)
            if phase < 0.3:  # Exploration Phase
                scale = 0.8 * self._dynamic_scaling(phase, memory, best_fitness) * local_exp_factor / np.sqrt(self.dim)
            elif phase < 0.7:  # Balanced Phase
                scale = 0.2 * self._fitness_variance(memory) * local_exp_factor / np.sqrt(self.dim)
            else:  # Exploitation Phase
                scale = 0.05 * self._fitness_variance(memory) / np.sqrt(self.dim)

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

            # Adjust population size and learning rate dynamically based on convergence speed and diversity
            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))

        return best_solution

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

    def _local_exploration_factor(self, func, point, lb, ub, samples=5):
        nearby_points = np.clip(point + np.random.uniform(-0.1, 0.1, size=(samples, self.dim)) * (ub - lb), lb, ub)
        fitness_values = np.array([func(p) for p in nearby_points])
        local_variance = np.std(fitness_values)
        return 1.0 + local_variance