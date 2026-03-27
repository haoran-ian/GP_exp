import numpy as np

class EnhancedSimulatedAnnealingWithRestarts:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        alpha = 0.9
        restarts = 5
        evaluations_per_restart = self.budget // restarts
        best_solution = None
        best_value = float('inf')

        for _ in range(restarts):
            current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            current_value = func(current_solution)
            evaluations = 1

            # Define dual annealing schedules
            schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / evaluations_per_restart)
            schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / evaluations_per_restart))

            # Enhanced Simulated Annealing with Restarts Loop
            while evaluations < evaluations_per_restart:
                # Toggle between two annealing schedules
                if evaluations % 3 == 0:
                    T = schedule_A(evaluations)
                else:
                    T = schedule_B(evaluations)

                # Dynamic Neighborhood Scaling with fitness guidance
                scale_factor = 1 - ((current_value - best_value) / (1 + abs(best_value)))
                scale = (func.bounds.ub - func.bounds.lb) * scale_factor * (1 - (evaluations / evaluations_per_restart) ** (1 / alpha))
                perturbation = np.random.normal(0, scale / 5, self.dim)
                candidate_solution = current_solution + perturbation
                candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
                candidate_value = func(candidate_solution)
                evaluations += 1

                # Metropolis criterion
                if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / T):
                    current_solution = candidate_solution
                    current_value = candidate_value

                    # Update the best solution found in this restart
                    if current_value < best_value:
                        best_solution = np.copy(current_solution)
                        best_value = current_value
        
        return best_solution, best_value