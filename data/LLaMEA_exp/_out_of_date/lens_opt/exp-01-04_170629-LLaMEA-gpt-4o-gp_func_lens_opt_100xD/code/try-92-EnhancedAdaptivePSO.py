import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30  # Standard number of particles in PSO
        self.inertia_weight = 0.9  # Start with a higher inertia weight
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.gaussian_scale = 0.1
        self.min_inertia_weight = 0.4  # Minimum inertia weight
        self.inertia_weight_decay = 0.99  # Decay rate for inertia weight

    def levy_flight(self, L):
        return np.random.normal(0, 1, self.dim) / (np.abs(np.random.normal(0, 1)) ** (1 / L))

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        # Initialize positions and velocities
        positions = np.random.uniform(lower_bound, upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])

        # Initialize global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.num_particles  # Initial evaluations

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])

                # Use Lévy flights instead of Gaussian perturbation for exploration
                levy_perturbation = self.gaussian_scale * self.levy_flight(1.5)

                # Adjust velocity with Lévy perturbation
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_velocity +
                                 social_velocity +
                                 levy_perturbation)

                # Update positions
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)

                # Evaluate new position
                score = func(positions[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                # Dynamic inertia weight adjustment
                self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight * self.inertia_weight_decay)

                # Stop if budget is exhausted
                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score