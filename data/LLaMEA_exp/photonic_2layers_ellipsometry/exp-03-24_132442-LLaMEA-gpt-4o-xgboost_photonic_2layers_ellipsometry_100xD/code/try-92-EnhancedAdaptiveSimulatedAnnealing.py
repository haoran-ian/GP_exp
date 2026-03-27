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

        # Define dual annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Enhanced Adaptive Simulated Annealing Loop
        while evaluations < self.budget:
            if evaluations % 3 == 0:
                T = schedule_A(evaluations)
            else:
                T = schedule_B(evaluations)
            
            # Dynamic scaling of perturbation with a decay factor
            decay_factor = 1 - (evaluations / self.budget)
            scale = (func.bounds.ub - func.bounds.lb) * decay_factor ** (1 / alpha)
            perturbation_factor = 1 + 0.3 * np.sin(2 * np.pi * evaluations / self.budget)  # Further modified perturbation
            perturbation = np.random.normal(0, scale / (5 * perturbation_factor), self.dim)
            candidate_solution = current_solution + perturbation
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_solution)
            evaluations += 1
            
            # Stochastic acceptance threshold with dynamic probability adjustment
            acceptance_probability = np.exp((current_value - candidate_value) / T)
            stochastic_threshold = acceptance_probability * (0.5 + 0.5 * np.random.rand())
            
            # Metropolis criterion with stochastic threshold
            if candidate_value < current_value or np.random.rand() < stochastic_threshold:
                current_solution = candidate_solution
                current_value = candidate_value

                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value
        
        return best_solution, best_value