import numpy as np

class AdaptiveLearningSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight_start = 0.9
        self.inertia_weight_end = 0.4
        self.cognitive_coeff_base = 2.5
        self.social_coeff_base = 1.5
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
            mean_score = np.mean(self.pbest_scores)
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

                # Adaptive learning factor influenced by swarm performance
                self.cognitive_coeff = self.cognitive_coeff_base * (self.pbest_scores[i] / mean_score)
                self.social_coeff = self.social_coeff_base * (self.gbest_score / mean_score)
                
                cognitive_velocity = self.cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.social_coeff * r2 * (self.gbest_position - self.positions[i])
                
                # Non-linear decay for inertia weight
                self.inertia_weight = self.inertia_weight_end + \
                    (self.inertia_weight_start - self.inertia_weight_end) * \
                    np.exp(-5 * eval_count / self.budget)
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] 
                                      + cognitive_velocity + social_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], bounds.lb, bounds.ub)
            
            # Diversity-enhancing mechanism with adaptive threshold
            diversity = np.std(self.positions, axis=0)
            adaptive_threshold = 0.1 * (bounds.ub - bounds.lb).mean() * np.exp(-eval_count / self.budget)
            if np.mean(diversity) < adaptive_threshold:
                self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

            if eval_count >= self.budget:
                break

        return self.gbest_position, self.gbest_score