import numpy as np

class EnhancedMemorySimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.92  # Slightly increased for control and exploration
        self.memory_size = 10  # Memory to store high-quality solutions
        self.memory = []
        
    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        current_value = func(current_solution)
        best_solution = np.copy(current_solution)
        best_value = current_value

        # Store initial solution in memory
        self.memory.append((current_solution, current_value))

        evaluations = 1

        # Define dual annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Enhanced Simulated Annealing Loop
        while evaluations < self.budget:
            # Toggle between two annealing schedules
            if evaluations % 3 == 0:  # Changed from 2 to 3 for more schedule variation
                T = schedule_A(evaluations)
            else:
                T = schedule_B(evaluations)
            
            # Adaptive Neighborhood Scaling
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / self.alpha))
            perturbation_factor = 1 + 0.25 * np.sin(2 * np.pi * evaluations / self.budget)
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
            
            # Update memory with high-quality solutions
            if len(self.memory) < self.memory_size or candidate_value < max(self.memory, key=lambda x: x[1])[1]:
                if len(self.memory) >= self.memory_size:
                    self.memory.pop(-1)
                self.memory.append((candidate_solution, candidate_value))
                self.memory.sort(key=lambda x: x[1])
            
            # Occasionally introduce a solution from memory to guide search
            if evaluations % 10 == 0:
                memory_solution, _ = self.memory[np.random.randint(0, len(self.memory))]
                current_solution = memory_solution

        return best_solution, best_value