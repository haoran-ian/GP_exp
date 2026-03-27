import numpy as np

class AdaptiveSpiralSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize the spiral parameters
        center = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        best_value = float('inf')
        best_position = center.copy()
        
        # Initial spiral parameters
        angle_increment = 0.1  # Spiral angle increment
        radius = 0.1  # Initial radius of the spiral

        while self.evaluations < self.budget:
            for i in range(1, self.budget - self.evaluations + 1):
                # Calculate spiral position
                angle = i * angle_increment
                displacement = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle)
                ] + [0] * (self.dim - 2))  # Only first two dimensions spiral
                
                # Wrap around if dimensions > 2
                spiral_position = center + displacement[:self.dim]
                
                # Ensure within bounds
                spiral_position = np.clip(spiral_position, func.bounds.lb, func.bounds.ub)
                
                # Evaluate the function
                value = func(spiral_position)
                self.evaluations += 1
                
                # Update best found position
                if value < best_value:
                    best_value = value
                    best_position = spiral_position
                    center = spiral_position
                    radius *= 0.9  # Reduce radius to focus the search
                    break  # Restart spiral with new center
            else:
                # Increase spiral radius if no better solution is found
                radius *= 1.1

        return best_position, best_value