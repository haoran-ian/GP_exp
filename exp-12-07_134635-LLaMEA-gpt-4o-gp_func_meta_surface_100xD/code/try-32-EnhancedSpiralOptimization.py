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
        learning_rate = 0.1  # New learning rate parameter

        # Dynamic factors
        phase_switch_threshold = self.budget // 3  # Switch phases after one-third of evaluations
        contraction_factor = 0.8
        expansion_factor = 1.1

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

            # Check if the candidate is better
            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                # Adaptive adjustments
                radius *= contraction_factor  # More aggressive contraction for better solutions
                inertia_weight = max(0.4, inertia_weight - learning_rate * 0.05)  # Decrease inertia for exploitation
            else:
                # Increase exploration
                radius *= expansion_factor  # More aggressive expansion for worse solutions
                inertia_weight = min(0.9, inertia_weight + learning_rate * 0.01)  # Increase inertia for exploration

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

            # Adaptive phase switching
            if evaluations % phase_switch_threshold == 0:
                # Switch contraction and expansion factors for a refined search
                contraction_factor *= 0.95  # Slow down contraction to refine search
                expansion_factor *= 1.05  # Slow down expansion for controlled exploration

        return best_position, best_value