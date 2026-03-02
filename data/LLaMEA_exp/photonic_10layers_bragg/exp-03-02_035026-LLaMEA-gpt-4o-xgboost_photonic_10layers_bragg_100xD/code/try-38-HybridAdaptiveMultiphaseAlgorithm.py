import numpy as np

class HybridAdaptiveMultiphaseAlgorithm:
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
            
            if phase < 0.3:  # Exploration Phase with hybrid mutation
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.6 * self._fitness_variance(best_fitness) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate, hybrid=True)
            elif phase < 0.7:  # Balanced Phase with adaptive learning rate
                learning_rate = 0.1 + 0.4 * phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.2 * self._fitness_variance(best_fitness) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate)
            else:  # Exploitation Phase with focused mutation
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.05 * self._fitness_variance(best_fitness) / np.sqrt(self.dim), population_size=population_size, learning_rate=learning_rate, hybrid=False)
            
            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
            
            # Adjust population size dynamically based on convergence speed
            population_size = max(5, int(20 - 15 * phase + 5 * (np.std(candidate_fitness) / abs(best_fitness))))
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, population_size, learning_rate, hybrid=True):
        # Generates new candidate solutions around a center point with hybrid mutation
        if hybrid:
            perturbations = np.random.normal(0, scale, size=(population_size, self.dim)) * learning_rate
        else:
            perturbations = np.random.uniform(-scale, scale, size=(population_size, self.dim)) * learning_rate
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)