import numpy as np

class MultiScheduleSimulatedAnnealing:
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
        
        # Define multiple annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))
        schedule_C = lambda evals: T_initial / (1 + evals)

        # Multi-Schedule Simulated Annealing Loop
        while evaluations < self.budget:
            # Rotate through different annealing schedules
            if evaluations % 4 == 0:
                T = schedule_A(evaluations)
            elif evaluations % 4 == 1:
                T = schedule_B(evaluations)
            else:
                T = schedule_C(evaluations)
            
            # Adaptive Neighborhood Scaling with dynamic perturbation
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** 0.5)
            perturbation = np.random.normal(0, scale * (0.3 + 0.7 * np.sin(np.pi * evaluations / self.budget)), self.dim)
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