import numpy as np

class AdaptiveMultiphasePerturbationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1

        # Initial random search to improve exploration
        num_initial_random_searches = min(10, self.budget - evaluations)
        initial_solutions = np.random.uniform(lb, ub, size=(num_initial_random_searches, self.dim))
        initial_fitness = np.array([func(sol) for sol in initial_solutions])
        evaluations += num_initial_random_searches

        min_idx = np.argmin(initial_fitness)
        if initial_fitness[min_idx] < best_fitness:
            best_solution = initial_solutions[min_idx]
            best_fitness = initial_fitness[min_idx]

        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.5 * self._fitness_variance(best_fitness))
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.2 * self._fitness_variance(best_fitness))
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.05 * self._fitness_variance(best_fitness))

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale):
        # Generates new candidate solutions around a center point
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)