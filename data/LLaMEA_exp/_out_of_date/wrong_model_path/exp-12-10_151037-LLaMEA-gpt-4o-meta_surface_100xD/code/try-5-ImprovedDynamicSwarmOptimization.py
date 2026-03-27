import numpy as np

class ImprovedDynamicSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.initial_cognitive_coeff = 2.0
        self.final_cognitive_coeff = 0.5
        self.initial_social_coeff = 0.5
        self.final_social_coeff = 2.0
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

                # Adaptive inertia weight based on diversity
                diversity = np.std(self.positions, axis=0).mean()
                inertia_weight = self.final_inertia_weight + (self.initial_inertia_weight - self.final_inertia_weight) * (1 - eval_count/self.budget)
                
                # Adaptive cognitive coefficient
                cognitive_coeff = self.final_cognitive_coeff + (self.initial_cognitive_coeff - self.final_cognitive_coeff) * diversity / (diversity + 1e-9)
                
                # Adaptive social coefficient
                social_coeff = self.final_social_coeff + (self.initial_social_coeff - self.final_social_coeff) * (1 - diversity / (diversity + 1e-9))

                cognitive_velocity = cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = social_coeff * r2 * (self.gbest_position - self.positions[i])
                
                self.velocities[i] = (inertia_weight * self.velocities[i] 
                                      + cognitive_velocity + social_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], bounds.lb, bounds.ub)

            if eval_count >= self.budget:
                break

        return self.gbest_position, self.gbest_score