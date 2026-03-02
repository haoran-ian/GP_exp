import numpy as np

class EnhancedAdaptiveMultiphaseAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        population_size = 10

        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.7 * self._fitness_variance(best_fitness), population_size=population_size)
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.3 * self._fitness_variance(best_fitness), population_size=population_size)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.1 * self._fitness_variance(best_fitness), population_size=population_size)
            
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
            
            # Adjust population size dynamically based on convergence speed
            population_size = max(5, int(15 - 10 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, population_size):
        # Generates new candidate solutions around a center point
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 5)