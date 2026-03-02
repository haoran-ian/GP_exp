import numpy as np

class EnhancedAdaptiveMemoryAlgorithm:
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

            if phase < 0.4:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.9 * self._dynamic_scaling(phase, memory, best_fitness) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate)
            elif phase < 0.8:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.3 * self._fitness_variance(memory) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.1 * self._fitness_variance(memory) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            # Adaptive neighborhood exploration
            for idx, fitness in enumerate(candidate_fitness):
                if fitness < best_fitness:
                    best_solution = candidate_solutions[idx]
                    best_fitness = fitness
                    break

            memory.append(best_fitness)
            if len(memory) > 15:
                memory.pop(0)

            # Adjust population size and learning rate dynamically based on convergence speed and diversity
            population_size = max(5, int(25 - 20 * phase + 10 * (np.std(candidate_fitness) / abs(best_fitness))))
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