import numpy as np

class AdvancedSpiralOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        evaluations = 0
        best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_position)
        evaluations += 1

        radius = (self.upper_bound - self.lower_bound) / 2
        inertia_weight = 0.9
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        stagnation_counter = 0  # New: Counter for stagnation

        while evaluations < self.budget:
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)

            candidate_value = func(candidate_position)
            evaluations += 1

            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                radius *= 0.8
                inertia_weight = max(0.4, inertia_weight - 0.05)
                stagnation_counter = 0  # Reset on improvement
            else:
                radius *= 1.05  # Adjusted: Softer expansion
                inertia_weight = min(0.9, inertia_weight + 0.02)  # Adjusted: Faster increase
                stagnation_counter += 1

            if stagnation_counter > 10:  # New: Adaptive change after stagnation
                radius *= 1.2  # New: Increase exploration
                inertia_weight = 0.9  # New: Reset inertia

            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value