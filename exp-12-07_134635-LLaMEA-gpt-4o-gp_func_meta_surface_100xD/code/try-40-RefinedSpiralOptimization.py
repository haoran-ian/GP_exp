import numpy as np

class RefinedSpiralOptimization:
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
            # Adjust learning rate and inertia dynamically
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))
            inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)

            # Introduce spiral variance by generating multiple candidates
            num_candidates = 3
            candidate_positions = []
            candidate_values = []

            for _ in range(num_candidates):
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.random.normal(0, 1, self.dim)
                direction /= np.linalg.norm(direction)
                candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate_position)
                candidate_positions.append(candidate_position)
                candidate_values.append(candidate_value)
                evaluations += 1
            
            # Select the best candidate
            best_candidate_idx = np.argmin(candidate_values)
            candidate_position = candidate_positions[best_candidate_idx]
            candidate_value = candidate_values[best_candidate_idx]

            # Check if the candidate is better
            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
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