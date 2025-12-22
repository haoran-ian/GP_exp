import numpy as np

class EnhancedAdaptiveRandomSearchV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        # Memory to store past successful moves
        self.memory = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5
            reduction_factor = 0.3

            while self.evaluations < self.budget:
                # Utilize memory of successful steps for exploration
                if len(self.memory) > 0 and np.random.rand() < 0.5:
                    memory_vector = np.random.choice(self.memory)
                    candidate = np.clip(current_best + memory_vector, lb, ub)
                else:
                    candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    step_vector = candidate - current_best
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.3
                    # Store successful steps in memory
                    self.memory.append(step_vector)
                    if len(self.memory) > 10:  # Limit memory size
                        self.memory.pop(0)
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.85

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0
                    reduction_factor *= 1.1

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            adaptive_step_size = (ub - lb) / 10

        return best_solution