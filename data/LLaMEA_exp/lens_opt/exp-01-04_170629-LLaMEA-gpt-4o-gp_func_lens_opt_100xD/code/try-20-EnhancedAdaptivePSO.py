import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_param = 2.0
        self.social_param = 2.0
        self.gaussian_scale = 0.1
        self.levy_alpha = 1.5  # Lévy flight alpha parameter

    def levy_flight(self, scale):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1/self.levy_alpha)
        return scale * step

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)

        positions = np.random.uniform(lower_bound, upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])

                gaussian_perturbation = self.gaussian_scale * np.random.normal(0, 1, self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_velocity +
                                 social_velocity +
                                 gaussian_perturbation)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)

                if np.random.rand() < 0.1:  # Apply Lévy flight with some probability
                    positions[i] += self.levy_flight(scale=0.1 * (upper_bound - lower_bound))

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

                if evaluations >= self.budget:
                    break

            self.inertia_weight = max(self.inertia_weight - (0.5 / self.budget) * evaluations, self.inertia_weight_min)

        return global_best_position, global_best_score