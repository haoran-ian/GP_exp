import numpy as np

class EnhancedAdaptiveThresholdAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        population_size = 10
        adaptive_threshold = 0.1  # Initial threshold for dynamic adaptation

        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            if phase < 0.3:  # Exploration Phase
                scale = 0.5 * self._fitness_variance(best_fitness) / np.sqrt(self.dim)
            elif phase < 0.7:  # Balanced Phase
                scale = 0.3 * self._fitness_variance(best_fitness) / np.sqrt(self.dim)
            else:  # Exploitation Phase
                scale = 0.1 * self._fitness_variance(best_fitness) / np.sqrt(self.dim)
            
            candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size)
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness - adaptive_threshold * np.abs(best_fitness):
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
                adaptive_threshold *= 0.9  # Reduce threshold when improvement is found
            else:
                adaptive_threshold *= 1.1  # Increase threshold when no improvement
            
            # Adjust population size dynamically based on convergence speed
            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, population_size):
        # Generates new candidate solutions around a center point
        perturbations = np.random.uniform(-scale, scale, size=(population_size, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)