import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9  # Starting inertia weight
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.initial_gaussian_scale = 0.1
        self.final_gaussian_scale = 0.01

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (abs(v) ** (1 / beta))
        return step

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

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update inertia weight dynamically
                inertia_weight_dynamic = self.inertia_weight - (0.5 * evaluations / self.budget)

                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])

                # Calculate current gaussian scale
                gaussian_scale_dynamic = self.initial_gaussian_scale - (
                    (self.initial_gaussian_scale - self.final_gaussian_scale) * (evaluations / self.budget))

                # Apply Levy flight perturbation
                levy_perturbation = self.levy_flight()

                # Adjust velocity with Levy perturbation
                velocities[i] = (inertia_weight_dynamic * velocities[i] +
                                 cognitive_velocity +
                                 social_velocity +
                                 gaussian_scale_dynamic * levy_perturbation)

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

                # Stop if budget is exhausted
                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score