import numpy as np

class AdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight_start = 0.9
        self.inertia_weight_end = 0.4
        self.cognitive_coeff_start = 2.5
        self.cognitive_coeff_end = 1.0
        self.social_coeff_start = 1.5
        self.social_coeff_end = 0.5
        self.local_coeff = 1.0
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = None
        self.pbest_positions = None
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf

    def __call__(self, func):
        bounds = func.bounds
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.pbest_positions = self.positions.copy()

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                eval_count += 1
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()

                r1, r2, r3 = np.random.random(), np.random.random(), np.random.random()
                self.cognitive_coeff = self.cognitive_coeff_start - (self.cognitive_coeff_start - self.cognitive_coeff_end) * (eval_count / self.budget)
                self.social_coeff = self.social_coeff_start - (self.social_coeff_start - self.social_coeff_end) * (eval_count / self.budget)
                inertia_weight = self.inertia_weight_start - (self.inertia_weight_start - self.inertia_weight_end) * (eval_count / self.budget)

                neighbors_size = int(max(2, 5 - 3 * (eval_count / self.budget)))
                neighbors = np.random.choice(self.population_size, size=neighbors_size, replace=False)
                local_best_position = self.pbest_positions[neighbors[np.argmin(self.pbest_scores[neighbors])]]

                cognitive_velocity = self.cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.social_coeff * r2 * (self.gbest_position - self.positions[i])
                local_velocity = self.local_coeff * r3 * (local_best_position - self.positions[i])

                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity + local_velocity
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], bounds.lb, bounds.ub)

            diversity = np.std(self.positions, axis=0)
            if np.mean(diversity) < 0.1 * (bounds.ub - bounds.lb).mean():
                self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

            if eval_count >= self.budget:
                break

        return self.gbest_position, self.gbest_score