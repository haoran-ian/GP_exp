import numpy as np

class AdaptiveDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, max(10, dim * 2))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_function_calls = 0
        self.velocity_clamp = 0.1 * (self.upper_bound - self.lower_bound)

    def __call__(self, func):
        c1_initial, c2_initial = 2.5, 2.5
        c1_final, c2_final = 0.5, 0.5
        while self.fitness_function_calls < self.budget:
            for i in range(self.population_size):
                if self.fitness_function_calls >= self.budget:
                    break
                fitness = func(self.particles[i])
                self.fitness_function_calls += 1
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.particles[i]

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.particles[i]

            progress_ratio = self.fitness_function_calls / self.budget
            inertia_weight = 0.9 - 0.5 * (progress_ratio ** 2)  # Non-linear inertia adjustment

            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i])
                self.velocities[i] += social_constant * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Diversity-driven selection strategy
            diversity_measure = np.std(self.particles, axis=0).mean()
            if diversity_measure < 0.1:  # Threshold for diversity
                mutation_strength = (self.upper_bound - self.lower_bound) * 0.05 * (1 - progress_ratio)
                mutation = np.random.uniform(-mutation_strength, mutation_strength, self.particles.shape)
                self.particles += mutation
                self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            # Improved elitist strategy to preserve better solutions
            elite_size = max(1, self.population_size // 10)
            elite_idxs = np.argsort(self.personal_best_scores)[:elite_size]
            elite_positions = self.personal_best_positions[elite_idxs]
            for i in range(elite_size):
                self.particles[elite_idxs[i]] = elite_positions[i]

        return self.global_best_position