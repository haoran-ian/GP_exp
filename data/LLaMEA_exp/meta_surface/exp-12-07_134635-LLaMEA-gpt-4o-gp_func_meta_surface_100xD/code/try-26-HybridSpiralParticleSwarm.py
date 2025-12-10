import numpy as np

class HybridSpiralParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize variables
        evaluations = 0
        n_particles = 10
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (n_particles, self.dim))
        velocities = np.zeros((n_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_values = np.array([func(p) for p in personal_best_positions])
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]
        evaluations += n_particles

        # Spiral and PSO parameters
        inertia_weight = 0.9
        learning_rate = 0.1
        cognitive_coeff = 2.0
        social_coeff = 2.0

        while evaluations < self.budget:
            for i in range(n_particles):
                # Spiral dynamic adjustment
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.random.normal(0, 1, self.dim)
                direction /= np.linalg.norm(direction)
                spiral_adjustment = inertia_weight * np.cos(angle) * direction

                # Particle Swarm Optimization velocity update
                r1, r2 = np.random.rand(2)
                cognitive_component = cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                social_component = social_coeff * r2 * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

                # Update position
                candidate_position = positions[i] + velocities[i] + spiral_adjustment
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)

                # Evaluate the candidate solution
                candidate_value = func(candidate_position)
                evaluations += 1

                # Update personal and global bests
                if candidate_value < personal_best_values[i]:
                    personal_best_positions[i] = candidate_position
                    personal_best_values[i] = candidate_value

                    if candidate_value < global_best_value:
                        global_best_position = candidate_position
                        global_best_value = candidate_value

            # Adaptive inertia weight adjustment
            inertia_weight = max(0.4, inertia_weight - learning_rate * 0.01)

        return global_best_position, global_best_value