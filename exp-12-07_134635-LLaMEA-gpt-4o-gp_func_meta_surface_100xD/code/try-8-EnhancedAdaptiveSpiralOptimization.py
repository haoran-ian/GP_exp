import numpy as np

class EnhancedAdaptiveSpiralOptimization:
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

        spiral_angle = np.pi / 4  # Initial angle for spiral
        angle_modulation_rate = 0.1  # Rate for changing the spiral angle

        while evaluations < self.budget:
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            spiral_step = inertia_weight * radius * np.cos(spiral_angle) * direction
            candidate_position = best_position + spiral_step

            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
            candidate_value = func(candidate_position)
            evaluations += 1

            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                radius *= 0.85
                inertia_weight = max(0.4, inertia_weight - 0.05)
                spiral_angle -= angle_modulation_rate  # Reduce angle for finer exploitation
            else:
                radius *= 1.15
                inertia_weight = min(0.9, inertia_weight + 0.01)
                spiral_angle += angle_modulation_rate  # Increase angle for broader exploration

            radius = max(radius, min_radius)
            radius = min(radius, max_radius)
            spiral_angle = np.clip(spiral_angle, 0, np.pi / 2)  # Keep angle within sensible limits

        return best_position, best_value