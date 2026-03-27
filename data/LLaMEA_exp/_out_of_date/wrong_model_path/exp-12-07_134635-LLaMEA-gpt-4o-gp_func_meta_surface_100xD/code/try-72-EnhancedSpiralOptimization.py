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
        inertia_weight = 0.9  # Initial inertia weight
        min_radius = (self.upper_bound - self.lower_bound) * 0.01
        max_radius = (self.upper_bound - self.lower_bound) * 0.5
        base_learning_rate = 0.1  # Base learning rate

        # Additional parameters for memory-based dynamics
        recent_best_positions = [best_position]
        memory_size = 5

        while evaluations < self.budget:
            # Adjust learning rate dynamically based on evaluations ratio
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))

            # Generate a new candidate solution on the spiral
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            memory_adjustment = np.mean(recent_best_positions, axis=0)
            candidate_position = best_position + inertia_weight * radius * np.cos(angle) * direction + 0.1 * (memory_adjustment - best_position)

            # Ensure candidate is within bounds
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)

            # Evaluate the candidate solution
            candidate_value = func(candidate_position)
            evaluations += 1

            # Check if the candidate is better
            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                radius *= 0.8  # More aggressive contraction for better solutions
                inertia_weight = max(0.4, inertia_weight - learning_rate * 0.05)  # Decrease inertia for exploitation
                recent_best_positions.append(best_position)  # Update memory
                if len(recent_best_positions) > memory_size:
                    recent_best_positions.pop(0)
            else:
                # Increase exploration
                radius *= 1.1  # More aggressive expansion for worse solutions
                inertia_weight = min(0.9, inertia_weight + learning_rate * 0.01)  # Increase inertia for exploration

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value