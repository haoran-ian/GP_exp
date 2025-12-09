import numpy as np

class RefinedSpiralOptimization:
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

        initial_radius = (self.upper_bound - self.lower_bound) / 2
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        base_learning_rate = 0.1
        inertia_weight = 0.9

        # Chaos-based randomization
        chaos_factor = np.random.uniform()

        while evaluations < self.budget:
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))

            # Generate a new candidate using chaotic map
            angle = 2 * np.pi * chaos_factor
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            radius = initial_radius * np.sqrt(1 - evaluations / self.budget)

            candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction

            # Dynamic boundary control
            candidate_position = np.clip(candidate_position, 
                                         self.lower_bound + (self.upper_bound - self.lower_bound) * 0.1, 
                                         self.upper_bound - (self.upper_bound - self.lower_bound) * 0.1)

            candidate_value = func(candidate_position)
            evaluations += 1

            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                initial_radius *= 0.8
                inertia_weight = max(0.4, inertia_weight - learning_rate * 0.05)
            else:
                initial_radius *= 1.1
                inertia_weight = min(0.9, inertia_weight + learning_rate * 0.01)

            initial_radius = max(initial_radius, min_radius)
            initial_radius = min(initial_radius, max_radius)

            # Update chaos factor for next iteration
            chaos_factor = (chaos_factor * 3.9 * (1 - chaos_factor)) % 1

        return best_position, best_value