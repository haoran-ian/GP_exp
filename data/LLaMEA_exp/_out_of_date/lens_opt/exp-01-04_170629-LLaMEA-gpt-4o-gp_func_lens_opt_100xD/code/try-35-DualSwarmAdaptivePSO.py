import numpy as np

class DualSwarmAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.gaussian_scale = 0.1
        self.inertia_damping = 0.99  # New dynamic inertia

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        # Initialize positions and velocities for two swarms
        positions_1 = np.random.uniform(lower_bound, upper_bound, (self.num_particles // 2, self.dim))
        positions_2 = np.random.uniform(lower_bound, upper_bound, (self.num_particles // 2, self.dim))
        velocities_1 = np.random.uniform(-1, 1, (self.num_particles // 2, self.dim))
        velocities_2 = np.random.uniform(-1, 1, (self.num_particles // 2, self.dim))
        personal_best_positions_1 = np.copy(positions_1)
        personal_best_positions_2 = np.copy(positions_2)
        personal_best_scores_1 = np.array([func(p) for p in positions_1])
        personal_best_scores_2 = np.array([func(p) for p in positions_2])

        # Initialize global bests
        global_best_idx_1 = np.argmin(personal_best_scores_1)
        global_best_position_1 = personal_best_positions_1[global_best_idx_1]
        global_best_score_1 = personal_best_scores_1[global_best_idx_1]

        global_best_idx_2 = np.argmin(personal_best_scores_2)
        global_best_position_2 = personal_best_positions_2[global_best_idx_2]
        global_best_score_2 = personal_best_scores_2[global_best_idx_2]

        evaluations = self.num_particles  # Initial evaluations

        while evaluations < self.budget:
            for swarm in [(positions_1, velocities_1, personal_best_positions_1, personal_best_scores_1, global_best_position_1, global_best_score_1),
                         (positions_2, velocities_2, personal_best_positions_2, personal_best_scores_2, global_best_position_2, global_best_score_2)]:
                positions, velocities, personal_best_positions, personal_best_scores, global_best_position, global_best_score = swarm

                for i in range(positions.shape[0]):
                    # Update velocities
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                    social_velocity = self.social_param * r2 * (global_best_position - positions[i])

                    # Apply Gaussian perturbation to enhance exploration
                    gaussian_perturbation = self.gaussian_scale * np.random.normal(0, 1, self.dim)

                    # Adjust velocity with Gaussian perturbation
                    velocities[i] = (self.inertia_weight * velocities[i] +
                                     cognitive_velocity +
                                     social_velocity +
                                     gaussian_perturbation)

                    # Update positions
                    positions[i] += velocities[i]
                    positions[i] = np.clip(positions[i], lower_bound, upper_bound)

                    # Evaluate new position
                    score = func(positions[i])
                    evaluations += 1

                    # Update personal bests
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = positions[i]

                    # Update global bests
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i]

                    # Update swarm tuple
                    swarm = (positions, velocities, personal_best_positions, personal_best_scores, global_best_position, global_best_score)

                    # Stop if budget is exhausted
                    if evaluations >= self.budget:
                        break

            # Update the main global best based on both swarms
            if global_best_score_1 < global_best_score_2:
                global_best_position_main = global_best_position_1
                global_best_score_main = global_best_score_1
            else:
                global_best_position_main = global_best_position_2
                global_best_score_main = global_best_score_2

            # Dynamically adjust the inertia weight
            self.inertia_weight *= self.inertia_damping

        return global_best_position_main, global_best_score_main