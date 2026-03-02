import numpy as np

class EnhancedDynamicAdaptiveAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        memory = []
        
        # Adaptive parameters
        init_pop_size = 15
        min_pop_size = 5
        init_lr = 0.2
        min_lr = 0.01
        feedback_factor = 0.5

        while evaluations < self.budget:
            phase = evaluations / self.budget
            population_size = max(min_pop_size, int(init_pop_size - (init_pop_size - min_pop_size) * phase))
            learning_rate = max(min_lr, init_lr * np.exp(-feedback_factor * phase))
            
            if phase < 0.3:  # Exploration Phase
                scale = 0.8 * self._dynamic_scaling(phase, memory, best_fitness) / np.sqrt(self.dim)
            elif phase < 0.7:  # Balanced Phase
                scale = 0.3 * self._fitness_variance(memory) / np.sqrt(self.dim)
            else:  # Exploitation Phase
                scale = 0.1 * self._fitness_variance(memory) / np.sqrt(self.dim)

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