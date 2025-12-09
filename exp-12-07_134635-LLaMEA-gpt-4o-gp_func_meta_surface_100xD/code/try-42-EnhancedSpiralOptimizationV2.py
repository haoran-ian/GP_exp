import numpy as np

class EnhancedSpiralOptimizationV2:
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
        base_learning_rate = 0.05  # Reduced base learning rate for finer adjustments

        # Chaotic map initialization (Logistic map)
        logistic_map = np.random.uniform()
        chaos_factor = 3.9  # Chaotic parameter for logistic map

        while evaluations < self.budget:
            # Chaotic map update
            logistic_map = chaos_factor * logistic_map * (1 - logistic_map)

            # Adjust learning rate dynamically based on evaluations ratio
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))

            # Update inertia weight dynamically
            inertia_weight = 0.9 - (0.5 * (evaluations / self.budget))

            # Generate a new candidate solution on the spiral
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction

            # Influence of chaotic map
            candidate_position += chaos_factor * (logistic_map - 0.5) * (candidate_position - best_position)

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
                radius *= 0.85  # Aggressive contraction for better solutions
            else:
                # Increase exploration
                radius *= 1.05  # Gentle expansion for worse solutions

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value