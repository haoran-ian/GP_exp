import numpy as np

class DynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Initial swarm size
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive (personal) weight
        self.c2 = 1.5  # Social (global) weight

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        eval_count = 0
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Evaluate the current position of each particle
                current_score = func(self.particles[i])
                eval_count += 1
                if eval_count >= self.budget:
                    break

                # Update personal best if current position is better
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.particles[i]

                # Update global best if current position is better
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.particles[i]

            # Update velocity and position
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]
                # Apply bounds
                self.particles[i] = np.clip(self.particles[i], lb, ub)

        return self.global_best_position, self.global_best_score