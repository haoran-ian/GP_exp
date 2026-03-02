import numpy as np

class EnhancedAdaptiveDiversityAlgorithm:
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
        diversity_factor = 0.5

        while evaluations < self.budget:
            phase = evaluations / self.budget

            # Dynamically adjust diversity factor based on phase and fitness variance
            diversity_factor = self._dynamic_diversity(phase, memory, best_fitness)

            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=diversity_factor * 0.8 / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate)
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=diversity_factor * 0.4 / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=diversity_factor * 0.1 / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate)

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

    def _dynamic_diversity(self, phase, memory, best_fitness):
        if not memory:
            return 0.5
        fitness_variance = np.std(memory) / abs(best_fitness) if best_fitness != 0 else 0
        return 0.5 + 0.5 * fitness_variance * (1 - phase)