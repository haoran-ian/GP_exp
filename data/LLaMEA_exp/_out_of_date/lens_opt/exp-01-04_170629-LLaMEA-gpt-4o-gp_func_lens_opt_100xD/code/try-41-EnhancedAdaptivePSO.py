import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30  # Standard number of particles in PSO
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.gaussian_scale = 0.1
        self.stagnation_threshold = 20  # Steps without improvement to trigger restart

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
        best_score_history = [global_best_score]
        stagnation_counter = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Calculate inertia weight adaptively
                inertia_weight = self.inertia_weight_max - (
                    (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
                )

                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])
                
                # Apply Gaussian perturbation to enhance exploration
                gaussian_perturbation = self.gaussian_scale * np.random.normal(0, 1, self.dim)
                
                # Adjust velocity with Gaussian perturbation
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_velocity +
                                 social_velocity +
                                 gaussian_perturbation)

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
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Stop if budget is exhausted
                if evaluations >= self.budget:
                    break

            # Implement random restart if stagnation is detected
            if stagnation_counter >= self.stagnation_threshold:
                restart_idx = np.random.randint(self.num_particles)
                positions[restart_idx] = np.random.uniform(lower_bound, upper_bound, self.dim)
                velocities[restart_idx] = np.random.uniform(-1, 1, self.dim)
                stagnation_counter = 0

        return global_best_position, global_best_score