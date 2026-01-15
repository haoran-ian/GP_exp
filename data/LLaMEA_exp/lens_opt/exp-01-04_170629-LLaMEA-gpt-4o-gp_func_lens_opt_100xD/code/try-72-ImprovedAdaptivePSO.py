import numpy as np

class ImprovedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_param = 2.0
        self.social_param = 2.0
        self.perturbation_scale = 0.2

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
            inertia_weight = (self.inertia_weight_max - self.inertia_weight_min) * \
                             ((self.budget - evaluations) / self.budget) + self.inertia_weight_min
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_param * r1 * (personal_best_positions[i] - positions[i])
                social_velocity = self.social_param * r2 * (global_best_position - positions[i])
                
                stochastic_perturbation = self.perturbation_scale * np.random.standard_normal(self.dim)

                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_velocity +
                                 social_velocity +
                                 stochastic_perturbation)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)

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

        return global_best_position, global_best_score