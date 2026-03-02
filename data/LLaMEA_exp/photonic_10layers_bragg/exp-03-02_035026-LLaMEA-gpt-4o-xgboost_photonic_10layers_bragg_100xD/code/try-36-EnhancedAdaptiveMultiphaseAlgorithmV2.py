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

        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            if phase < 0.3:  # Exploration Phase
                scale = self._adaptive_scale(phase, best_fitness)
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size)
            elif phase < 0.7:  # Balanced Phase
                scale = self._adaptive_scale(phase, best_fitness)
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size)
            else:  # Exploitation Phase
                scale = self._adaptive_scale(phase, best_fitness)
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale, population_size)
            
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]

            # Diversity-preserving adjustment
            diversity_factor = np.std(candidate_fitness) / abs(best_fitness)
            population_size = max(5, int(20 - 15 * phase + 10 * diversity_factor))
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, population_size):
        # Generates new candidate solutions around a center point
        perturbations = np.random.normal(0, scale, size=(population_size, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _adaptive_scale(self, phase, best_fitness):
        # Adjusts the scale based on phase and fitness variance
        base_scale = max(0.05, 0.5 / np.sqrt(self.dim))
        if phase < 0.3:
            return base_scale * (1 + phase) * 0.6 * self._fitness_variance(best_fitness)
        elif phase < 0.7:
            return base_scale * (1 + phase) * 0.2 * self._fitness_variance(best_fitness)
        else:
            return base_scale * (1 + phase) * 0.05 * self._fitness_variance(best_fitness)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)