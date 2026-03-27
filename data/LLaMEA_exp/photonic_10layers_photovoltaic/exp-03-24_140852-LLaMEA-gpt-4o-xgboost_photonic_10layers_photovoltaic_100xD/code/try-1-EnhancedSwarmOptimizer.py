import numpy as np

class EnhancedSwarmOptimizer:
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
        self.w = 0.9  # Start with a higher inertia weight
        self.c1 = 2.0  # Increase cognitive weight
        self.c2 = 2.0  # Increase social weight
        self.local_search_intensity = 0.1  # Intensity of local search

    def adaptive_parameters(self, eval_progress):
        # Adaptively adjust inertia weight
        self.w = 0.9 - (0.5 * eval_progress)
        # Optionally adjust cognitive and social weights if needed
        self.c1 = 2.0 + (0.5 * eval_progress)
        self.c2 = 2.0 - (0.5 * eval_progress)

    def local_search(self, particle):
        # Simple local search around the particle's current position
        perturbation = self.local_search_intensity * np.random.uniform(-1, 1, self.dim)
        return np.clip(particle + perturbation, 0, 1)

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

                # Local search intensification
                local_candidate = self.local_search(self.particles[i])
                local_score = func(local_candidate)
                eval_count += 1

                if local_score < current_score:
                    current_score = local_score
                    self.particles[i] = local_candidate

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

            # Adaptive parameter tuning
            eval_progress = eval_count / self.budget
            self.adaptive_parameters(eval_progress)

        return self.global_best_position, self.global_best_score