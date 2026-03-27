import numpy as np

class EnhancedAdaptiveMultiSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.num_swarms = 3
        self.swarm_size = self.population_size // self.num_swarms
        self.inertia_weight_start = 0.9
        self.inertia_weight_end = 0.4
        self.cognitive_coeff_start = 2.5
        self.cognitive_coeff_end = 1.0
        self.social_coeff = 1.5
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = None
        self.pbest_positions = None
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_positions = [None] * self.num_swarms
        self.gbest_scores = np.full(self.num_swarms, np.inf)

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

                swarm_id = i // self.swarm_size
                if score < self.gbest_scores[swarm_id]:
                    self.gbest_scores[swarm_id] = score
                    self.gbest_positions[swarm_id] = self.positions[i].copy()

                r1, r2 = np.random.random(), np.random.random()
                self.cognitive_coeff = (
                    self.cognitive_coeff_start - (self.cognitive_coeff_start - self.cognitive_coeff_end) * (eval_count / self.budget)
                )
                cognitive_velocity = self.cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                
                # Choose gbest from current or neighboring swarm for inter-swarm communication
                neighbor_swarm_id = (swarm_id + 1) % self.num_swarms
                if np.random.random() < 0.5:
                    gbest_position = self.gbest_positions[swarm_id]
                else:
                    gbest_position = self.gbest_positions[neighbor_swarm_id]
                    
                social_velocity = self.social_coeff * r2 * (gbest_position - self.positions[i])

                self.inertia_weight = (
                    self.inertia_weight_start - (self.inertia_weight_start - self.inertia_weight_end) * (eval_count / self.budget)
                )
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] 
                                      + cognitive_velocity + social_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], bounds.lb, bounds.ub)

            diversity = np.std(self.positions, axis=0)
            if np.mean(diversity) < 0.1 * (bounds.ub - bounds.lb).mean():
                self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

            if eval_count >= self.budget:
                break

        best_swarm_id = np.argmin(self.gbest_scores)
        return self.gbest_positions[best_swarm_id], self.gbest_scores[best_swarm_id]