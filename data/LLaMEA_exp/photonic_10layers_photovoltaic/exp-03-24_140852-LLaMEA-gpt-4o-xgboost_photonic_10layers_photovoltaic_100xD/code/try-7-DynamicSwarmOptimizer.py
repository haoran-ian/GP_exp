import numpy as np

class DynamicSwarmOptimizer:
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
        self.w = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.v_max = 0.2  # Maximum velocity for clamping

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        eval_count = 0
        leadership_change_interval = self.budget // 10

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

            if eval_count % leadership_change_interval == 0:
                self.global_best_position = np.mean(self.personal_best_positions, axis=0)

            self.w = self.w_min + (0.9 - self.w_min) * (self.budget - eval_count) / self.budget

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

                mutation = np.random.normal(0, 0.1, self.dim)
                self.particles[i] += self.velocities[i] + mutation
                self.particles[i] = np.clip(self.particles[i], lb, ub)

                # Local search around the global best
                if eval_count < self.budget - self.population_size:
                    search_variation = np.random.normal(0, 0.01, self.dim)
                    potential_solution = self.global_best_position + search_variation
                    potential_solution = np.clip(potential_solution, lb, ub)
                    potential_score = func(potential_solution)
                    eval_count += 1

                    if potential_score < self.global_best_score:
                        self.global_best_score = potential_score
                        self.global_best_position = potential_solution

        return self.global_best_position, self.global_best_score