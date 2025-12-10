import numpy as np

class DynamicAdaptiveSpiralOptimization:
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
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        learning_rate_decay = 0.99
        momentum = 0.9
        velocity = np.zeros(self.dim)

        while evaluations < self.budget:
            # Dynamic learning rate decreases over time
            learning_rate = (0.1 / (1 + (evaluations / self.budget) * learning_rate_decay))
            
            # Generate a candidate solution using spiral dynamics
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            velocity = momentum * velocity + learning_rate * radius * np.cos(angle) * direction
            candidate_position = best_position + velocity
            
            # Boundary check
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
            candidate_value = func(candidate_position)
            evaluations += 1

            # Update best position if a better solution is found
            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                radius *= 0.7  # Contraction
            else:
                radius *= 1.2  # Expansion

            # Adjust momentum dynamically based on the number of evaluations
            momentum = 0.9 * (1 - evaluations / self.budget) + 0.1

            # Ensure radius stays within the specified bounds
            radius = np.clip(radius, min_radius, max_radius)

        return best_position, best_value