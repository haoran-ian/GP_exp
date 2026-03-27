import numpy as np

class HybridAdaptiveSimulatedAnnealing:
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

        # Define dual annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Hybrid Adaptive Simulated Annealing Loop
        while evaluations < self.budget:
            # Toggle between two annealing schedules
            if evaluations % 3 == 0:
                T = schedule_A(evaluations)
            else:
                T = schedule_B(evaluations)
            
            # Adaptive Neighborhood Scaling with stochastic perturbation
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
            perturbation_factor = 1 + 0.2 * np.sin(2 * np.pi * evaluations / self.budget)
            perturbation = np.random.normal(0, scale / (5 * perturbation_factor), self.dim)
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
            
            # Integrate Differential Evolution for additional exploration
            if evaluations % 5 == 0:
                for _ in range(3):  # Perform multiple DE steps
                    idxs = np.random.choice(np.arange(self.dim), 3, replace=False)
                    donor_vector = current_solution[idxs[0]] + 0.8 * (current_solution[idxs[1]] - current_solution[idxs[2]])
                    trial_vector = np.where(np.random.rand(self.dim) < 0.5, donor_vector, current_solution)
                    trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)
                    trial_value = func(trial_vector)
                    evaluations += 1
                    
                    if trial_value < current_value:
                        current_solution = trial_vector
                        current_value = trial_value

                        if current_value < best_value:
                            best_solution = np.copy(current_solution)
                            best_value = current_value
        
        return best_solution, best_value