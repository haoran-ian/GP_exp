import numpy as np

class HybridMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.num_swarms = 3
        self.swarms = [np.random.rand(self.population_size, dim) for _ in range(self.num_swarms)]
        self.velocities = [np.random.rand(self.population_size, dim) * 0.1 for _ in range(self.num_swarms)]
        self.personal_best_positions = [np.copy(swarm) for swarm in self.swarms]
        self.personal_best_scores = [np.full(self.population_size, np.inf) for _ in range(self.num_swarms)]
        self.global_best_positions = [None for _ in range(self.num_swarms)]
        self.global_best_scores = [np.inf for _ in range(self.num_swarms)]
        self.w = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.v_max = 0.2
        self.mutation_rate = 0.1

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        eval_count = 0
        mutation_adaptation_interval = self.budget // 20

        while eval_count < self.budget:
            for swarm_index in range(self.num_swarms):
                for i in range(self.population_size):
                    current_score = func(self.swarms[swarm_index][i])
                    eval_count += 1
                    if eval_count >= self.budget:
                        break

                    if current_score < self.personal_best_scores[swarm_index][i]:
                        self.personal_best_scores[swarm_index][i] = current_score
                        self.personal_best_positions[swarm_index][i] = self.swarms[swarm_index][i]

                    if current_score < self.global_best_scores[swarm_index]:
                        self.global_best_scores[swarm_index] = current_score
                        self.global_best_positions[swarm_index] = self.swarms[swarm_index][i]

                if eval_count % mutation_adaptation_interval == 0:
                    self.mutation_rate *= 0.95

                self.w = self.w_min + (0.9 - self.w_min) * (self.budget - eval_count) / self.budget

                for i in range(self.population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive_component = self.c1 * r1 * (self.personal_best_positions[swarm_index][i] - self.swarms[swarm_index][i])
                    social_component = self.c2 * r2 * (self.global_best_positions[swarm_index] - self.swarms[swarm_index][i])
                    self.velocities[swarm_index][i] = self.w * self.velocities[swarm_index][i] + cognitive_component + social_component
                    
                    self.velocities[swarm_index][i] = np.clip(self.velocities[swarm_index][i], -self.v_max, self.v_max)

                    mutation = np.random.normal(0, self.mutation_rate, self.dim)
                    self.swarms[swarm_index][i] += self.velocities[swarm_index][i] + mutation
                    self.swarms[swarm_index][i] = np.clip(self.swarms[swarm_index][i], lb, ub)

        # Select the best solution among all swarms
        best_global_score = min(self.global_best_scores)
        best_swarm_index = self.global_best_scores.index(best_global_score)
        return self.global_best_positions[best_swarm_index], best_global_score