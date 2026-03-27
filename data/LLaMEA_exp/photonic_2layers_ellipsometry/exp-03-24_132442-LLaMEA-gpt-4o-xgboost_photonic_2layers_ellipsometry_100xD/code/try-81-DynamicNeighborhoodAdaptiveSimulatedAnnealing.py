import numpy as np

class DynamicNeighborhoodAdaptiveSimulatedAnnealing:
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

        # Define dual annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Dynamic Neighborhood Scaling with ruggedness assessment
        def landscape_ruggedness(evals, candidate_value, current_value):
            # A simple measure of ruggedness using value difference
            return np.abs(candidate_value - current_value) / (1 + evals / self.budget)

        # Improved Adaptive Simulated Annealing Loop
        while evaluations < self.budget:
            # Toggle between two annealing schedules
            if evaluations % 3 == 0:
                T = schedule_A(evaluations)
            else:
                T = schedule_B(evaluations)
            
            # Dynamic Neighborhood Scaling
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
            perturbation = np.random.normal(0, scale / 5, self.dim)
            candidate_solution = current_solution + perturbation
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_solution)
            evaluations += 1
            
            # Adjust scale based on landscape ruggedness
            ruggedness = landscape_ruggedness(evaluations, candidate_value, current_value)
            adaptive_scale = 0.5 + 0.5 * np.tanh(ruggedness)
            perturbation *= adaptive_scale
            
            # Metropolis criterion
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / T):
                current_solution = candidate_solution
                current_value = candidate_value

                # Update the best solution found
                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value
        
        return best_solution, best_value