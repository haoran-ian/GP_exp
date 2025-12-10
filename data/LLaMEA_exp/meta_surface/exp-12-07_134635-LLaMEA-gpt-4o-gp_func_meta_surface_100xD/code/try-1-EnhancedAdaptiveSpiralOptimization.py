import numpy as np

class EnhancedAdaptiveSpiralOptimization:
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
        angle_step = np.pi / 8  # Changed angle step for finer granularity

        while evaluations < self.budget:
            # Generate a new candidate solution on the spiral
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            candidate_position = best_position + radius * np.cos(angle) * direction

            # Ensure candidate is within bounds and adjust if necessary
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)

            # Evaluate the candidate solution
            candidate_value = func(candidate_position)
            evaluations += 1

            # Check if the candidate is better
            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                # Adaptive adjustments
                radius *= 0.85  # Slightly more aggressive contraction
            else:
                # Increase exploration
                radius *= 1.05  # Slightly more conservative expansion

            # Ensure radius remains within sensible bounds
            radius = max(radius, (self.upper_bound - self.lower_bound) * 0.01)
            radius = min(radius, (self.upper_bound - self.lower_bound) * 0.5)

        return best_position, best_value