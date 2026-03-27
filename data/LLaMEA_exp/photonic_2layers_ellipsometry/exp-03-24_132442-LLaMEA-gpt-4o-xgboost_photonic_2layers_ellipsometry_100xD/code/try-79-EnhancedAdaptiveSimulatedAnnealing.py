import numpy as np

class EnhancedAdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        alpha = 0.92
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        current_value = func(current_solution)
        best_solution = np.copy(current_solution)
        best_value = current_value

        evaluations = 1

        # Define a dynamic cooling schedule
        dynamic_schedule = lambda evals: T_initial * np.exp(-2 * (evals / self.budget)**2)

        # Improved Adaptive Simulated Annealing Loop
        while evaluations < self.budget:
            T = dynamic_schedule(evaluations)
            
            # Adaptive Neighborhood Scaling with an additional perturbation factor
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
            perturbation_factor = 1 + 0.25 * np.sin(2 * np.pi * evaluations / self.budget)
            perturbation = np.random.normal(0, scale / (5 * perturbation_factor), self.dim)
            candidate_solution = current_solution + perturbation
            
            # Implementing boundary reflection
            candidate_solution = np.where(candidate_solution < func.bounds.lb, 
                                          func.bounds.lb + np.abs(candidate_solution - func.bounds.lb), candidate_solution)
            candidate_solution = np.where(candidate_solution > func.bounds.ub, 
                                          func.bounds.ub - np.abs(candidate_solution - func.bounds.ub), candidate_solution)

            candidate_value = func(candidate_solution)
            evaluations += 1
            
            # Metropolis criterion
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / T):
                current_solution = candidate_solution
                current_value = candidate_value

                # Update the best solution found
                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value
        
        return best_solution, best_value