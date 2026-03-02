import numpy as np

class EnhancedAdaptiveMultiphasePerturbationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1
        historical_best_fitness = best_fitness
        
        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            # Feedback mechanism based on rate of improvement
            improvement_rate = (historical_best_fitness - best_fitness) / historical_best_fitness
            scale_factor = max(0.05, 0.5 * improvement_rate)
            
            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.6 * scale_factor / np.sqrt(self.dim))
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.2 * scale_factor / np.sqrt(self.dim))
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.05 * scale_factor / np.sqrt(self.dim))
            
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
                historical_best_fitness = min(historical_best_fitness, best_fitness)
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale):
        # Generates new candidate solutions around a center point
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)