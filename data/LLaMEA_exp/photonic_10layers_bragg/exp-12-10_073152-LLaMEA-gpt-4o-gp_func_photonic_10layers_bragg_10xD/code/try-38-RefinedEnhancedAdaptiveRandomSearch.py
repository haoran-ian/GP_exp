import numpy as np

class RefinedEnhancedAdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.phase_length = 10  # Initial phase length

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        adaptive_step_size = (ub - lb) / 5  # Initial larger step size

        # Memory of past best solutions to aid in exploration
        memory = []
        memory_limit = 5

        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            phase_counter = 0

            while self.evaluations < self.budget and phase_counter < self.phase_length:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    adaptive_step_size *= 1.3  # Increased step size adjustment on improvement
                    phase_counter = 0  # Reset phase counter on improvement
                else:
                    adaptive_step_size *= 0.9  # Adjusted step size reduction on no improvement
                    phase_counter += 1  # Increase phase counter

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Store best solution found in this run into memory
            if len(memory) < memory_limit:
                memory.append(current_best)
            else:
                # Replace the oldest memory entry
                memory.pop(0)
                memory.append(current_best)

            # Multi-phase strategy: explore the memory solutions
            for mem_solution in memory:
                if self.evaluations >= self.budget:
                    break
                candidate = np.clip(mem_solution + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                if candidate_value < best_value:
                    best_solution = candidate
                    best_value = candidate_value

            # Dynamically adjust adaptive step size based on overall progress
            adaptive_step_size = (ub - lb) / (5 + 3 * (1 - best_value / float('inf')))  # Adaptive scaling based on progress
            self.phase_length = min(20, 5 + int(self.budget / (self.evaluations + 1)))  # Adjust phase length based on evaluations

        return best_solution