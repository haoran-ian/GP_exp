import numpy as np

class EnhancedSwarmOptimizer:
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
        self.neighborhood_size = 5
        self.adaptive_threshold = 0.2

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

            self.w = self.w_min + (self.w_init - self.w_min) * (self.global_best_score / np.min(self.personal_best_scores))

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component + np.random.normal(0, 0.01, self.dim)

                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

                neighborhood_best_position = self.find_neighborhood_best(i)
                if eval_count % 10 == 0 and np.random.rand() < self.adaptive_threshold:
                    informed_search = np.random.normal(neighborhood_best_position, 0.1, self.dim)
                    informed_search = np.clip(informed_search, lb, ub)
                    informed_search_score = func(informed_search)
                    eval_count += 1
                    if informed_search_score < self.global_best_score:
                        self.global_best_score = informed_search_score
                        self.global_best_position = informed_search

                adaptive_mutation = np.random.normal(0, np.random.uniform(0.05, 0.15) * exploration_exploitation_ratio[eval_count], self.dim)
                self.particles[i] += self.velocities[i] + adaptive_mutation
                
                self.particles[i] = np.clip(self.particles[i], lb, ub)

        return self.global_best_position, self.global_best_score

    def find_neighborhood_best(self, index):
        start = max(0, index - self.neighborhood_size // 2)
        end = min(self.population_size, index + self.neighborhood_size // 2 + 1)
        neighborhood = self.personal_best_scores[start:end]
        best_index = np.argmin(neighborhood) + start
        return self.personal_best_positions[best_index]