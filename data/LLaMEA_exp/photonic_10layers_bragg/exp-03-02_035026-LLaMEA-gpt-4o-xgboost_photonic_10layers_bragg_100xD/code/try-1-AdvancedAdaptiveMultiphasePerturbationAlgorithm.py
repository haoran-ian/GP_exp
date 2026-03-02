import numpy as np

class AdvancedAdaptiveMultiphasePerturbationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = np.random.uniform(lb, ub, size=self.dim)
        best_fitness = func(best_solution)
        evaluations = 1

        while evaluations < self.budget:
            phase = evaluations / self.budget
            
            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.5, directional_bias=True)
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.2, directional_bias=False)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.05, directional_bias=False, adaptive_mutation=True)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, directional_bias=False, adaptive_mutation=False):
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        
        if directional_bias:
            bias = np.random.uniform(low=0.1, high=1.0, size=self.dim)
            perturbations *= bias

        if adaptive_mutation:
            mutation_rate = 1.0 - (np.abs(center) / (ub - lb))
            perturbations *= mutation_rate

        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)