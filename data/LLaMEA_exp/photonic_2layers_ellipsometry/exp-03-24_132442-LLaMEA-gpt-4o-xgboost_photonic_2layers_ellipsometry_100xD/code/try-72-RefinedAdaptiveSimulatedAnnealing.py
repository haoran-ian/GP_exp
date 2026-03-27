import numpy as np

class RefinedAdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)

        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        alpha = 0.92
        learning_rate = 0.1  # Introduced dynamic learning rate
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        current_value = func(current_solution)
        best_solution = np.copy(current_solution)
        best_value = current_value
        solution_memory = [current_solution]  # Memory-based search strategy

        evaluations = 1

        # Define dual annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Refined Adaptive Simulated Annealing Loop
        while evaluations < self.budget:
            T = schedule_A(evaluations) if evaluations % 3 == 0 else schedule_B(evaluations)

            # Adaptive Neighborhood Scaling with additional perturbation factor
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
            perturbation_factor = 1 + 0.25 * np.sin(2 * np.pi * evaluations / self.budget)
            perturbation = np.random.normal(0, scale / (5 * perturbation_factor), self.dim)
            
            # Memory-based candidate proposal
            memory_influence = np.mean(solution_memory, axis=0)
            candidate_solution = current_solution + learning_rate * (memory_influence - current_solution) + perturbation
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_solution)
            evaluations += 1

            # Metropolis criterion
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / T):
                current_solution = candidate_solution
                current_value = candidate_value
                solution_memory.append(candidate_solution)  # Update memory

                # Update the best solution found
                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value

            # Dynamic adjustment of learning rate
            learning_rate *= 0.995

        return best_solution, best_value