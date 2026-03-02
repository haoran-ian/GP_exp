import numpy as np

class EnhancedVarianceGuidedPerturbationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        self.memory.append(best_fitness)
        evaluations = 1

        while evaluations < self.budget:
            phase = evaluations / self.budget

            if phase < 0.3:  # Exploration Phase
                scale_factor = 0.5
            elif phase < 0.7:  # Balanced Phase
                scale_factor = 0.2
            else:  # Exploitation Phase
                scale_factor = 0.05
            
            scale = scale_factor * self._adaptive_scale(best_fitness)
            candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale)
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
                self.memory.append(best_fitness)

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale):
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _adaptive_scale(self, current_best_fitness):
        variance_factor = np.var(self.memory[-min(5, len(self.memory)):])
        return max(0.1, np.abs(current_best_fitness) / 10) * (1 + variance_factor)