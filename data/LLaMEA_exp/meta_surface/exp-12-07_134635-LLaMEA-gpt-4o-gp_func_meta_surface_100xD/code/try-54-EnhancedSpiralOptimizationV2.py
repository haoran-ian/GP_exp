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
        base_learning_rate = 0.1  # Base learning rate

        while evaluations < self.budget:
            # Adjust learning rate dynamically based on evaluations ratio
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))

            # Generate multiple candidate solutions for better exploration
            num_candidates = 5
            candidates = []
            candidate_values = []

            for _ in range(num_candidates):
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.random.normal(0, 1, self.dim)
                direction /= np.linalg.norm(direction)
                candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction

                # Ensure candidate is within bounds
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate_position)
                evaluations += 1

                candidates.append(candidate_position)
                candidate_values.append(candidate_value)

                # Stop if budget exceeded
                if evaluations >= self.budget:
                    break

            # Select the best candidate from the generated ones
            min_value_index = np.argmin(candidate_values)
            best_candidate_position = candidates[min_value_index]
            best_candidate_value = candidate_values[min_value_index]

            # Check if the best candidate is an improvement
            if best_candidate_value < best_value:
                best_position = best_candidate_position
                best_value = best_candidate_value
                # Adaptive adjustments
                radius *= 0.8  # More aggressive contraction for better solutions
                inertia_weight = max(0.4, inertia_weight - learning_rate * 0.05)  # Decrease inertia for exploitation
            else:
                # Increase exploration
                radius *= 1.1  # More aggressive expansion for worse solutions
                inertia_weight = min(0.9, inertia_weight + learning_rate * 0.01)  # Increase inertia for exploration

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value