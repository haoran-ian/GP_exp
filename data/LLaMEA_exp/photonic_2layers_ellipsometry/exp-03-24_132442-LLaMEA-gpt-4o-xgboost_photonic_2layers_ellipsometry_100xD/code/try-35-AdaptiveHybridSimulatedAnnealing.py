import numpy as np

class AdaptiveHybridSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        current_value = func(current_solution)
        best_solution = np.copy(current_solution)
        best_value = current_value

        evaluations = 1

        # New nonlinear temperature decay
        def temp_schedule(evals, T_initial, T_final, max_evals):
            return T_final + (T_initial - T_final) * np.exp(-10 * (evals / max_evals)**2)
        
        # Adaptive Hybrid Simulated Annealing Loop
        while evaluations < self.budget:
            T = temp_schedule(evaluations, T_initial, T_final, self.budget)
            
            # Dynamic Neighborhood Scaling with improved perturbation
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** 2)
            perturbation_factor = 1 + 0.5 * np.cos(2 * np.pi * evaluations / self.budget)
            perturbation = np.random.normal(0, scale / (3 * perturbation_factor), self.dim)
            candidate_solution = current_solution + perturbation
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_solution)
            evaluations += 1
            
            # Metropolis criterion with annealing probability adjustment
            delta = current_value - candidate_value
            acceptance_probability = np.exp(delta / T) if delta < 0 else 1.0
            
            if candidate_value < current_value or np.random.rand() < acceptance_probability:
                current_solution = candidate_solution
                current_value = candidate_value

                # Update the best solution found
                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value
        
        return best_solution, best_value