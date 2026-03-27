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
        base_learning_rate = 0.1

        # New adaptive parameters
        adaptive_phase_switch = 0.5  # Controls phase change
        exploitation_weight = 0.6  # Determining factor for exploitation enhancement

        while evaluations < self.budget:
            # Dynamic adjustment of learning rate
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))

            # Generate a new candidate solution on the spiral
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)

            # Adaptive multi-phase dynamics
            phase_factor = 1 if evaluations < self.budget * adaptive_phase_switch else exploitation_weight
            candidate_position = best_position + phase_factor * inertia_weight * radius * np.cos(angle) * direction

            # Ensure candidate is within bounds
            candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)

            # Evaluate the candidate solution
            candidate_value = func(candidate_position)
            evaluations += 1

            # Check if the candidate is better
            if candidate_value < best_value:
                best_position = candidate_position
                best_value = candidate_value
                # Aggressive contraction for better solutions
                radius *= 0.7
                inertia_weight = max(0.3, inertia_weight - learning_rate * 0.06)  # Decrease inertia more significantly
            else:
                # Increase exploration
                radius *= 1.3  # More aggressive expansion for worse solutions
                inertia_weight = min(0.95, inertia_weight + learning_rate * 0.03)  # Increase inertia

            # Adaptive inertia and convergence pressure balance
            convergence_pressure = 1 - evaluations / self.budget
            inertia_weight *= (0.9 - convergence_pressure * 0.5)

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value