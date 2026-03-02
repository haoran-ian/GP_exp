import numpy as np

class EnhancedAdaptiveMultiphaseAlgorithmV2:
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

        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            if phase < 0.3:  # Exploration Phase
                scale = 0.6 * self._fitness_variance(best_fitness) / np.sqrt(self.dim)
            elif phase < 0.7:  # Balanced Phase
                scale = 0.2 * self._fitness_variance(best_fitness) / np.sqrt(self.dim)
            else:  # Exploitation Phase
                scale = 0.05 * self._fitness_variance(best_fitness) / np.sqrt(self.dim)
            
            candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size, learning_rate)
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
            
            # Adjust population size dynamically based on fitness spread and gradient information
            gradient_magnitude = np.linalg.norm(np.gradient(candidate_fitness))
            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness) + gradient_magnitude)))
            learning_rate = max(0.01, learning_rate * (1 - np.std(candidate_fitness) / abs(best_fitness)))
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate):
        # Generates new candidate solutions around a center point
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * learning_rate * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)