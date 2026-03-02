import numpy as np

class AdaptiveMultiphasePerturbationAlgorithmPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        self.history.append(best_fitness)
        evaluations = 1

        while evaluations < self.budget:
            phase = evaluations / self.budget

            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.6 * self._dynamic_scale())
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.3 * self._dynamic_scale())
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.1 * self._dynamic_scale())

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)

            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
                self.history.append(best_fitness)

        return best_solution

    def _generate_solutions(self, center, lb, ub, scale):
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _dynamic_scale(self):
        variance = np.var(self.history)
        return max(0.1, np.sqrt(variance + 0.01))