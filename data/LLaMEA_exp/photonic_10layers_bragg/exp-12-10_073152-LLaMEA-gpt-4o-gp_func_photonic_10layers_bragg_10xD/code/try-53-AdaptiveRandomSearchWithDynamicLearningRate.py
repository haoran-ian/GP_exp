import numpy as np

class AdaptiveRandomSearchWithDynamicLearningRate:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Initialize learning rate parameters
        initial_step_size = (ub - lb) / 5
        min_step_size = initial_step_size / 100
        step_size = initial_step_size
        learning_rate = 0.2

        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5

            while self.evaluations < self.budget:
                # Adaptive step size exploration
                candidate = np.clip(current_best + np.random.uniform(-step_size, step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    # Increase step size dynamically during improvement
                    step_size = min(step_size * (1 + learning_rate), initial_step_size)
                else:
                    no_improvement_count += 1
                    # Gradually reduce step size when no improvement
                    step_size = max(step_size * (1 - learning_rate), min_step_size)

                if no_improvement_count >= max_no_improvement:
                    step_size *= 0.5  # Reduce step size scale more significantly
                    no_improvement_count = 0  # Reset count

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Multi-start strategy: restart with a refined exploration
            step_size = initial_step_size / 2  # Restart with a reduced step size

        return best_solution