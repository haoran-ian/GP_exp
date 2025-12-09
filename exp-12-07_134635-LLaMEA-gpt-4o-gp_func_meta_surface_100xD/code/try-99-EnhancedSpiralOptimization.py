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
        base_learning_rate = 0.1

        memory = best_position
        chaotic_param = 0.7

        while evaluations < self.budget:
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))
            
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
                radius *= chaotic_param
                memory = candidate_position
                chaotic_param -= 0.01
            else:
                radius *= 1.3
                chaotic_param = chaotic_param * (1 + 0.05 * np.random.rand())

            convergence_pressure = 1 - evaluations / self.budget
            inertia_weight = (0.9 - convergence_pressure * 0.6) * inertia_weight

            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value