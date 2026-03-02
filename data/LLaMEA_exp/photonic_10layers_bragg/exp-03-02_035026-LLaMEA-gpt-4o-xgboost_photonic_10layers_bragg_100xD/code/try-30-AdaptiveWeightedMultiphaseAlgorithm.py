import numpy as np

class AdaptiveWeightedMultiphaseAlgorithm:
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
            
            # Adaptive weighting based on phase and fitness improvement
            exploration_weight = (0.7 - 0.6 * phase) 
            exploitation_weight = (0.2 + 0.6 * phase)
            fitness_variance = self._fitness_variance(best_fitness)
            
            candidate_solutions = self._generate_solutions(
                best_solution, lb, ub,
                scale=(exploration_weight * fitness_variance / np.sqrt(self.dim)) + (exploitation_weight * np.std(best_fitness)/10),
                population_size=population_size
            )
            
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
            
            # Adjust population size with more sensitivity to fitness spread
            population_size = max(5, int(20 - 15 * phase + 10 * (np.std(candidate_fitness) / abs(best_fitness))))
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, population_size):
        # Generates new candidate solutions around a center point
        perturbations = np.random.uniform(-scale, scale, size=(population_size, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)