import numpy as np

class AdaptiveSwarmOptimizationImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
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

                r1, r2 = np.random.random(), np.random.random()
                # Nonlinear cognitive coefficient decay
                cognitive_coeff = 2.5 * (1 - (eval_count / self.budget)**2) + 0.5 * ((eval_count / self.budget)**2)
                cognitive_velocity = cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = 1.5 * r2 * (self.gbest_position - self.positions[i])
                
                # Nonlinear inertia weight adjustment
                inertia_weight = 0.9 - (0.5 * ((eval_count / self.budget)**2))
                
                self.velocities[i] = (inertia_weight * self.velocities[i] 
                                      + cognitive_velocity + social_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], bounds.lb, bounds.ub)

            if eval_count >= self.budget:
                break

        return self.gbest_position, self.gbest_score