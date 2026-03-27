import numpy as np

class AdaptiveMutationParticleSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.w_init = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.v_max = 0.2
        self.exploration_boost_interval = self.budget // 10
        self.local_search_interval = self.budget // 20  # New parameter for local search

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        eval_count = 0
        exploration_exploitation_ratio = np.linspace(0.2, 0.8, self.budget)

        while eval_count < self.budget:
            for i in range(self.population_size):
                current_score = func(self.particles[i])
                eval_count += 1
                if eval_count >= self.budget:
                    break

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.particles[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.particles[i]

            if eval_count % self.local_search_interval == 0:
                neighborhood_size = 5
                for _ in range(neighborhood_size):
                    search_variation = np.random.normal(0, 0.05, self.dim)
                    local_solution = self.global_best_position + search_variation
                    local_solution = np.clip(local_solution, lb, ub)
                    local_score = func(local_solution)
                    eval_count += 1

                    if local_score < self.global_best_score:
                        self.global_best_score = local_score
                        self.global_best_position = local_solution

            self.w = self.w_min + (self.w_init - self.w_min) * (self.budget - eval_count) / self.budget

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                
                if eval_count % self.exploration_boost_interval == 0:
                    exploration_boost = np.random.normal(0, 0.5, self.dim)
                    self.particles[i] += exploration_boost
                else:
                    adaptive_mutation = np.random.normal(0, np.random.uniform(0.05, 0.15) * exploration_exploitation_ratio[eval_count], self.dim)
                    self.particles[i] += self.velocities[i] + adaptive_mutation
                
                self.particles[i] = np.clip(self.particles[i], lb, ub)

        return self.global_best_position, self.global_best_score