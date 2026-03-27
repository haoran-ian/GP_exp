import numpy as np

class MemoryEnhancedSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory = []

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

        # Define dynamic memory size
        memory_size = 5
        
        # Geometric temperature scaling
        schedule = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)

        # Simulated Annealing with Memory
        while evaluations < self.budget:
            T = schedule(evaluations)
            
            # Adaptive Neighborhood Scaling with additional dynamic perturbation factor
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
            perturbation_factor = 1 + 0.25 * np.sin(2 * np.pi * evaluations / self.budget)
            perturbation = np.random.normal(0, scale / (5 * perturbation_factor), self.dim)
            candidate_solution = current_solution + perturbation
            candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_solution)
            evaluations += 1
            
            # Metropolis criterion with memory improvement
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / T):
                current_solution = candidate_solution
                current_value = candidate_value

                # Update memory
                self.memory.append((current_solution, current_value))
                if len(self.memory) > memory_size:
                    self.memory.pop(0)

                # Update the best solution found
                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value
            
            # Leverage memory to escape local minima
            if evaluations % (self.budget // memory_size) == 0:
                memory_solutions = [sol_value[0] for sol_value in self.memory]
                memory_values = [sol_value[1] for sol_value in self.memory]
                min_value_index = np.argmin(memory_values)
                best_memory_solution = memory_solutions[min_value_index]
                if best_memory_solution is not None:
                    current_solution = np.copy(best_memory_solution)
                    current_value = memory_values[min_value_index]

        return best_solution, best_value