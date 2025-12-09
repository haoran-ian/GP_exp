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
        initial_learning_rate = 0.1  # Initial learning rate
        angle_increment = np.pi / 4  # Initial angle increment

        while evaluations < self.budget:
            # Generate a new candidate solution on the spiral
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            candidate_position = best_position + inertia_weight * radius * (np.cos(angle) + np.sin(angle_increment)) * direction

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
                radius *= 0.8  # More aggressive contraction for better solutions
                inertia_weight = max(0.4, inertia_weight - initial_learning_rate * 0.05)  # Decrease inertia for exploitation
                angle_increment *= 0.9  # Reduce angle increment for finer exploration
                
            else:
                # Increase exploration
                radius *= 1.1  # More aggressive expansion for worse solutions
                inertia_weight = min(0.9, inertia_weight + initial_learning_rate * 0.01)  # Increase inertia for exploration
                angle_increment *= 1.1  # Increase angle increment for broader exploration

            # Update learning rate dynamically based on performance
            initial_learning_rate = max(0.01, initial_learning_rate * (0.95 if candidate_value < best_value else 1.05))

            # Ensure radius and angle increment remain within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)
            angle_increment = min(np.pi, max(angle_increment, np.pi / 8))

        return best_position, best_value