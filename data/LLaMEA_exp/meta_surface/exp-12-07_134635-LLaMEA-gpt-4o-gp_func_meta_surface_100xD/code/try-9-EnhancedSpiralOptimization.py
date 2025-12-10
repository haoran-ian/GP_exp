import numpy as np

class EnhancedSpiralOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize variables
        evaluations = 0
        best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_position)
        evaluations += 1
        # Spiral parameters
        radius = (self.upper_bound - self.lower_bound) / 2
        inertia_weight = 0.9  # Initial inertia weight
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        learning_rate = 0.1  # Adaptive learning rate

        while evaluations < self.budget:
            # Generate a new candidate solution using opposition-based learning
            opposite_position = self.lower_bound + self.upper_bound - best_position
            candidate_position = best_position + inertia_weight * radius * np.random.normal(0, 1, self.dim)
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
            opposite_position = np.clip(opposite_position, self.lower_bound, self.upper_bound)

            # Evaluate both positions
            candidate_value = func(candidate_position)
            opposite_value = func(opposite_position)
            evaluations += 2

            # Select the better candidate
            if candidate_value < best_value and candidate_value < opposite_value:
                best_position = candidate_position
                best_value = candidate_value
                radius *= 0.8
                inertia_weight = max(0.4, inertia_weight - 0.05)
                learning_rate = min(0.2, learning_rate + 0.01)
            elif opposite_value < best_value:
                best_position = opposite_position
                best_value = opposite_value
                radius *= 0.8
                inertia_weight = max(0.4, inertia_weight - 0.05)
                learning_rate = min(0.2, learning_rate + 0.01)
            else:
                radius *= 1.1
                inertia_weight = min(0.9, inertia_weight + 0.01)
                learning_rate = max(0.05, learning_rate - 0.005)

            # Ensure radius stays within bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value