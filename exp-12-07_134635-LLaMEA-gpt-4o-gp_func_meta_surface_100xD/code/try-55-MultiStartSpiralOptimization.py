import numpy as np

class MultiStartSpiralOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        evaluations = 0
        global_best_position = None
        global_best_value = float('inf')

        # Parameters
        num_starts = 5  # Number of multi-start attempts
        radius_decay = 0.9
        inertia_weight_decay = 0.95
        base_learning_rate = 0.1
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        
        for _ in range(num_starts):
            # Initialize variables for each start
            best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            best_value = func(best_position)
            evaluations += 1
            radius = max_radius
            inertia_weight = 0.9
            
            # Optimization loop
            while evaluations < self.budget / num_starts:
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
                    radius *= radius_decay
                    inertia_weight *= inertia_weight_decay
                else:
                    radius /= radius_decay
                    inertia_weight = min(0.9, inertia_weight + learning_rate * 0.01)

                radius = max(radius, min_radius)
                radius = min(radius, max_radius)

            if best_value < global_best_value:
                global_best_position = best_position
                global_best_value = best_value

        return global_best_position, global_best_value