import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9  # Start with a high inertia weight
        self.final_inertia_weight = 0.4  # Reduce it adaptively to emphasize exploitation
        self.cognitive_param = 2.0
        self.social_param = 2.0
        self.initial_gaussian_scale = 0.1
        self.final_gaussian_scale = 0.01  # Reduce Gaussian perturbation as search progresses

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
                # Calculate dynamic inertia weight
                inertia_weight = self.inertia_weight - (self.inertia_weight - self.final_inertia_weight) * (
                    evaluations / self.budget)
                
                # Calculate dynamic Gaussian perturbation scale
                gaussian_scale = self.initial_gaussian_scale - (self.initial_gaussian_scale - self.final_gaussian_scale) * (
                    evaluations / self.budget)

                # Update velocities with dynamic inertia and Gaussian perturbation
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])
                
                gaussian_perturbation = gaussian_scale * np.random.normal(0, 1, self.dim)
                
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

                # Stop if budget is exhausted
                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score