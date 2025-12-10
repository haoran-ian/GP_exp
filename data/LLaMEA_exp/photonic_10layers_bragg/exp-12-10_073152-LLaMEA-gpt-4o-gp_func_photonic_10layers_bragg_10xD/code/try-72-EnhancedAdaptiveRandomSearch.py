import numpy as np

class EnhancedAdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        while self.evaluations < self.budget:
            # Memory of best solutions
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            memory = [current_best]
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5
            reduction_factor = 0.3
            learning_rate = 0.05  # Introduced learning rate

            while self.evaluations < self.budget:
                perturbation = np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim)
                candidate = np.clip(current_best + perturbation, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= (1 + learning_rate)  # Dynamic learning rate
                    memory.append(current_best)  # Memory update
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= (1 - learning_rate)

                if no_improvement_count >= max_no_improvement:
                    if memory:  # Restart strategy using memory of past best solutions
                        current_best = memory[np.random.randint(len(memory))]
                        current_best_value = func(current_best)
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0
                    reduction_factor *= 1.1

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            adaptive_step_size = (ub - lb) / 10
            learning_rate = 0.05  # Reset learning rate

        return best_solution