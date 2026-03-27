import numpy as np

class AdvancedSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        alpha = 0.9
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        current_value = func(current_solution)
        best_solution = np.copy(current_solution)
        best_value = current_value

        evaluations = 1

        # Define adaptive annealing schedule with dynamic adjustment
        schedule = lambda evals: T_initial * (T_final / T_initial) ** ((evals / self.budget) ** 2)

        while evaluations < self.budget:
            T = schedule(evaluations)
            
            # Reinforcement learning-inspired perturbation with adaptive scaling
            scale = (func.bounds.ub - func.bounds.lb) * (1 - np.exp(-evaluations / (self.budget * alpha)))
            perturbation_strength = 0.1 + np.abs(np.sin(2 * np.pi * evaluations / self.budget))
            perturbation = np.random.normal(0, scale * perturbation_strength, self.dim)
            candidate_solution = current_solution + perturbation
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_solution)
            evaluations += 1
            
            # Enhanced Metropolis criterion with dynamic acceptance
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / T):
                current_solution = candidate_solution
                current_value = candidate_value

                # Update the best solution found
                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value
        
        return best_solution, best_value