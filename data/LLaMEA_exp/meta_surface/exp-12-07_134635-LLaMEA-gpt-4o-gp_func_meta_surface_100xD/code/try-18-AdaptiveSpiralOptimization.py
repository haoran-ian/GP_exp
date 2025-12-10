import numpy as np

class AdaptiveSpiralOptimization:
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
        inertia_weight = 0.9
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        learning_rate = 0.1

        # Adaptive parameters
        improvement_rate = 0.0
        improvement_threshold = 0.05

        while evaluations < self.budget:
            # Generate a new candidate solution on the spiral
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction

            # Ensure candidate is within bounds
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)

            # Evaluate the candidate solution
            candidate_value = func(candidate_position)
            evaluations += 1

            # Calculate improvement rate
            if candidate_value < best_value:
                improvement_rate = (best_value - candidate_value) / abs(best_value)
                best_position = candidate_position
                best_value = candidate_value

            # Check if the candidate is better
            if improvement_rate > improvement_threshold:
                # Adaptive adjustments for significant improvement
                radius *= 0.7
                inertia_weight = max(0.4, inertia_weight - learning_rate * 0.1)
            else:
                # Expand search for slow improvement
                radius *= 1.05
                inertia_weight = min(0.9, inertia_weight + learning_rate * 0.02)

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value