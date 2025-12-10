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
        inertia_weight = 0.9
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        base_learning_rate = 0.1

        success_counter = 0  # Track successful updates

        while evaluations < self.budget:
            # Dynamic learning rate based on success
            dynamic_lr = base_learning_rate * (1 + success_counter / self.budget)

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
                success_counter += 1  # Increment success
                radius *= 0.75  # Contract radius
                inertia_weight = max(0.3, inertia_weight - dynamic_lr * 0.05)  # Decrease inertia
            else:
                success_counter = max(0, success_counter - 1)  # Decrease success counter
                radius *= 1.1  # Expand radius slightly for exploration
                inertia_weight = min(0.95, inertia_weight + dynamic_lr * 0.03)  # Increase inertia

            # Adjust inertia weight and convergence pressure
            convergence_pressure = 1 - evaluations / self.budget
            inertia_weight = (0.9 - convergence_pressure * 0.5) * inertia_weight

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value