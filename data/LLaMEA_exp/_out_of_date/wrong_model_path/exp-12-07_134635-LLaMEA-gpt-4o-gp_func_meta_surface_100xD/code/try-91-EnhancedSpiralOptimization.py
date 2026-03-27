import numpy as np

class EnhancedSpiralOptimization:
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

        # Adjusted learning rate strategy to dynamically vary with budget and performance
        initial_learning_rate = 0.2

        while evaluations < self.budget:
            learning_rate = initial_learning_rate * np.exp(-evaluations / (self.budget / 2))
            
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)

            # Adaptive boundary expansion
            dynamic_bound = self.lower_bound - (5.0 * evaluations / self.budget)
            candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction
            candidate_position = np.clip(candidate_position, dynamic_bound, self.upper_bound)

            candidate_value = func(candidate_position)
            evaluations += 1

            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                radius *= 0.65  # Slightly more aggressive contraction
                inertia_weight = max(0.3, inertia_weight - learning_rate * 0.07)
            else:
                radius *= 1.2
                inertia_weight = min(0.95, inertia_weight + learning_rate * 0.05)

            convergence_pressure = 1 - evaluations / self.budget
            inertia_weight = (0.9 - convergence_pressure * 0.6) * inertia_weight

            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value