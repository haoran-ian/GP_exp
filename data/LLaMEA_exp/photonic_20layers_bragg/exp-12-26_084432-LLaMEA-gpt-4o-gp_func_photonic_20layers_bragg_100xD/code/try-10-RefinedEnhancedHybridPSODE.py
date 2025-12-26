import numpy as np

class RefinedEnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.f_min = 0.4
        self.f_max = 0.9
        self.cr = 0.9

    def __call__(self, func):
        np.random.seed(42)
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        positions = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), 
                                       (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            self.inertia_weight = self.inertia_weight_min + \
                                  (0.9 - self.inertia_weight_min) * (1 - evaluations / self.budget)

            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_component * r1 * (personal_best_positions - positions) +
                          self.social_component * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, lower_bound, upper_bound)

            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                    if scores[i] < global_best_score:
                        global_best_score = scores[i]
                        global_best_position = positions[i]

            self.f = self.f_min + (self.f_max - self.f_min) * (evaluations / self.budget)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                x1, x2, x3 = positions[indices]
                mutant = np.clip(x1 + self.f * (x2 - x3), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, positions[i])
                trial_score = func(trial)
                evaluations += 1

                if trial_score < scores[i]:
                    positions[i] = trial
                    scores[i] = trial_score
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial

        return global_best_position, global_best_score