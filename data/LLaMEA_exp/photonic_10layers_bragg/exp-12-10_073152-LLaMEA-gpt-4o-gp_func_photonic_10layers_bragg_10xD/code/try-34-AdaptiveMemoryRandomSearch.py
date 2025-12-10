import numpy as np

class AdaptiveMemoryRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = []  # Memory to store promising solutions

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        while self.evaluations < self.budget:
            if not self.memory or np.random.rand() < 0.3:
                current_best = np.random.uniform(lb, ub)
            else:
                # Select a solution from memory with slight random perturbation
                mem_idx = np.random.randint(len(self.memory))
                current_best = np.clip(self.memory[mem_idx] + np.random.normal(0, 0.1, self.dim), lb, ub)
            
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5
            reduction_factor = 0.5

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.2
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.85

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Update memory with the current best known solution
            if len(self.memory) < 10:
                self.memory.append(current_best)
            else:
                # Replace worst memory entry if new solution is better
                worst_idx = np.argmax([func(sol) for sol in self.memory])
                if current_best_value < func(self.memory[worst_idx]):
                    self.memory[worst_idx] = current_best

            # Multi-start strategy: restart with reduced step size for refined exploration
            adaptive_step_size = (ub - lb) / 10

        return best_solution