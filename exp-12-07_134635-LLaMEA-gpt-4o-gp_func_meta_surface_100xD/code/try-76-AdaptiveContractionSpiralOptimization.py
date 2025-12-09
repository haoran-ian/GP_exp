import numpy as np

class AdaptiveContractionSpiralOptimization:
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
        base_learning_rate = 0.1  # Base learning rate

        # Initialize adaptive contraction control variables
        contraction_factor = 0.8
        expansion_factor = 1.1
        consecutive_bad_steps = 0

        while evaluations < self.budget:
            # Adjust learning rate dynamically based on evaluations ratio
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))

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
                radius *= contraction_factor  # Contract the spiral more for better solutions
                inertia_weight = max(0.4, inertia_weight - learning_rate * 0.05)  # Decrease inertia for exploitation
                consecutive_bad_steps = 0  # Reset the count of consecutive bad steps
            else:
                # Increase exploration
                radius *= expansion_factor  # Expand the spiral for worse solutions
                inertia_weight = min(0.9, inertia_weight + learning_rate * 0.01)  # Increase inertia for exploration
                consecutive_bad_steps += 1

            # Adjust contraction and expansion factors based on consecutive bad steps
            if consecutive_bad_steps > 5:
                contraction_factor = max(0.7, contraction_factor - 0.05)
                expansion_factor = min(1.2, expansion_factor + 0.05)
            else:
                contraction_factor = 0.8
                expansion_factor = 1.1

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value