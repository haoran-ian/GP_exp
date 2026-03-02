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

        while evaluations < self.budget:
            phase = evaluations / self.budget
            learning_rate = self._dynamic_learning_rate(phase)
            population_size = self._dynamic_population_size(phase)
            
            if phase < 0.3:  # Exploration Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.6 * self._fitness_variance(best_fitness) * learning_rate, count=population_size)
            elif phase < 0.7:  # Balanced Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.2 * self._fitness_variance(best_fitness) * learning_rate, count=population_size)
            else:  # Exploitation Phase
                candidate_solutions = self._generate_solutions(best_solution, lb, ub, scale=0.05 * self._fitness_variance(best_fitness) * learning_rate, count=population_size)

            candidate_fitness = np.array([func(sol) for sol in candidate_solutions])
            evaluations += len(candidate_solutions)
            
            min_idx = np.argmin(candidate_fitness)
            if candidate_fitness[min_idx] < best_fitness:
                best_solution = candidate_solutions[min_idx]
                best_fitness = candidate_fitness[min_idx]
        
        return best_solution
    
    def _generate_solutions(self, center, lb, ub, scale, count):
        # Generates new candidate solutions around a center point
        perturbations = np.random.uniform(-scale, scale, size=(count, self.dim))
        solutions = center + perturbations * (ub - lb)
        return np.clip(solutions, lb, ub)

    def _fitness_variance(self, best_fitness):
        return max(0.1, np.abs(best_fitness) / 10)

    def _dynamic_learning_rate(self, phase):
        # Dynamically adjusts the learning rate based on the current phase
        if phase < 0.3:
            return 1.0
        elif phase < 0.7:
            return 0.5
        else:
            return 0.1

    def _dynamic_population_size(self, phase):
        # Dynamically adjusts the population size based on the current phase
        if phase < 0.3:
            return 20
        elif phase < 0.7:
            return 10
        else:
            return 5