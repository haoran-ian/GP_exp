import numpy as np

class HybridSpiralSwarmOptimization:
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

        # Particle swarm parameters
        particle_velocity = np.random.uniform(-1, 1, self.dim)
        personal_best_position = best_position
        personal_best_value = best_value
        global_best_position = best_position
        global_best_value = best_value

        while evaluations < self.budget:
            # Adjust learning rate dynamically based on evaluations ratio
            learning_rate = base_learning_rate * (1 - (evaluations / self.budget))

            # Generate a new candidate solution on the spiral
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.random.normal(0, 1, self.dim)
            direction /= np.linalg.norm(direction)
            spiral_position = personal_best_position + inertia_weight * radius * np.cos(angle) * direction

            # Ensure spiral position is within bounds
            spiral_position = np.clip(spiral_position, self.lower_bound, self.upper_bound)

            # Update particle velocity and position
            r1, r2 = np.random.rand(), np.random.rand()
            particle_velocity = (inertia_weight * particle_velocity
                                + r1 * (personal_best_position - spiral_position)
                                + r2 * (global_best_position - spiral_position))
            particle_position = spiral_position + particle_velocity

            # Ensure particle position is within bounds
            particle_position = np.clip(particle_position, self.lower_bound, self.upper_bound)

            # Evaluate the candidate solution
            candidate_value = func(particle_position)
            evaluations += 1

            # Check if the candidate is better
            if candidate_value < best_value:
                best_position = particle_position
                best_value = candidate_value

                # Update global best
                if candidate_value < global_best_value:
                    global_best_position = particle_position
                    global_best_value = candidate_value

                # Update personal best
                if candidate_value < personal_best_value:
                    personal_best_position = particle_position
                    personal_best_value = candidate_value

                # Adaptive adjustments
                radius *= 0.8  # More aggressive contraction for better solutions
                inertia_weight = max(0.4, inertia_weight - learning_rate * 0.05)  # Decrease inertia for exploitation
            else:
                # Increase exploration
                radius *= 1.1  # More aggressive expansion for worse solutions
                inertia_weight = min(0.9, inertia_weight + learning_rate * 0.01)  # Increase inertia for exploration

            # Ensure radius remains within sensible bounds
            radius = max(radius, min_radius)
            radius = min(radius, max_radius)

        return best_position, best_value