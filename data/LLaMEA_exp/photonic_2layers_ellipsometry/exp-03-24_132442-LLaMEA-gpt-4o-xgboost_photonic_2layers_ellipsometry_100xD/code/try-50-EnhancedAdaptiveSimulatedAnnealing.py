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
        alpha = 0.8  # Slightly changed alpha for more rapid cooling
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        current_value = func(current_solution)
        best_solution = np.copy(current_solution)
        best_value = current_value

        evaluations = 1

        # Dual annealing schedules with an added dynamic rescaling
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Enhanced Adaptive Simulated Annealing Loop
        while evaluations < self.budget:
            # Toggle between two annealing schedules
            if evaluations % 4 == 0:  # Changed to 4 for more schedule variation
                T = schedule_A(evaluations) * np.cos(evaluations / self.budget * np.pi/2)
            else:
                T = schedule_B(evaluations) * np.sin(evaluations / self.budget * np.pi/2)
            
            # Adaptive Neighborhood Scaling with multiple perturbation factors
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
            perturbation_factor_1 = 1 + 0.25 * np.sin(2 * np.pi * evaluations / self.budget)
            perturbation_factor_2 = 0.5 + 0.5 * np.cos(2 * np.pi * evaluations / self.budget)
            perturbation = np.random.normal(0, scale / (5 * perturbation_factor_1 * perturbation_factor_2), self.dim)
            candidate_solution = current_solution + perturbation
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
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