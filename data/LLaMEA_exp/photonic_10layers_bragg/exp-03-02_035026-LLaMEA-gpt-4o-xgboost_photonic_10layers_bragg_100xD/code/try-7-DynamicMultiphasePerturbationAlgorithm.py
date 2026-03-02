import numpy as np

class DynamicMultiphasePerturbationAlgorithm:
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
                scale = 0.5 * self._adaptive_scale(best_solution, best_fitness, func)
            elif phase < 0.7:  # Balanced Phase
                scale = 0.2 * self._adaptive_scale(best_solution, best_fitness, func)
            else:  # Exploitation Phase
                scale = 0.05 * self._adaptive_scale(best_solution, best_fitness, func)

            candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale)
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale):
        perturbations = np.random.uniform(-scale, scale, size=(10, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _adaptive_scale(self, solution, current_fitness, func):
        # Estimate local gradient
        epsilon = 1e-8
        gradient_est = np.zeros(self.dim)
        for i in range(self.dim):
            perturbed_solution = np.copy(solution)
            perturbed_solution[i] += epsilon
            gradient_est[i] = (func(perturbed_solution) - current_fitness) / epsilon

        local_search_intensity = np.linalg.norm(gradient_est) / (np.abs(current_fitness) + 1e-8)
        scaled_variance = max(0.1, np.abs(current_fitness) / 10)
        return scaled_variance / (1 + local_search_intensity)