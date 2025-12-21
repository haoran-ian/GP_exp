import numpy as np

class EnhancedLevyAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight_start = 0.9
        self.inertia_weight_end = 0.4
        self.cognitive_coeff_start = 2.5
        self.cognitive_coeff_end = 1.0
        self.social_coeff = 1.5
        self.alpha = 0.01  # Levy flight step size
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = None
        self.pbest_positions = None
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf

    def levy_flight(self):
        # Levy flight calculation using Mantegna's algorithm
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v) ** (1 / beta)
        return step

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

                r1, r2 = np.random.random(), np.random.random()
                self.cognitive_coeff = (
                    self.cognitive_coeff_start - (self.cognitive_coeff_start - self.cognitive_coeff_end) * (eval_count / self.budget)
                )
                cognitive_velocity = self.cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.social_coeff * r2 * (self.gbest_position - self.positions[i])
                
                self.inertia_weight = (
                    self.inertia_weight_start - (self.inertia_weight_start - self.inertia_weight_end) * (eval_count / self.budget)
                )
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] 
                                      + cognitive_velocity + social_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], bounds.lb, bounds.ub)

            # Levy flight application to improve exploration
            if eval_count < self.budget:
                for i in range(self.population_size):
                    if np.random.rand() < 0.3:  # 30% chance to apply Levy flight
                        levy_step = self.alpha * self.levy_flight()
                        self.positions[i] = np.clip(self.positions[i] + levy_step, bounds.lb, bounds.ub)

            # Diversity-enhancing mechanism
            diversity = np.std(self.positions, axis=0)
            if np.mean(diversity) < 0.1 * (bounds.ub - bounds.lb).mean():
                self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

            if eval_count >= self.budget:
                break

        return self.gbest_position, self.gbest_score