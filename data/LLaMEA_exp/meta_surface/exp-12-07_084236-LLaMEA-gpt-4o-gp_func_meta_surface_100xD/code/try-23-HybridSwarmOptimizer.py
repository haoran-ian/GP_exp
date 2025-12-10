import numpy as np

class HybridSwarmOptimizer:
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
        self.velocity_clamp = 0.1 * (self.upper_bound - self.lower_bound)  # Adaptive velocity clamping
        self.chaotic_sequence = np.array([np.sin(i) for i in range(self.budget)])  # Simple chaotic sequence

    def __call__(self, func):
        c1_initial, c2_initial = 2.5, 2.5
        c1_final, c2_final = 0.5, 0.5
        inertia_weight_min, inertia_weight_max = 0.4, 0.9
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
            inertia_weight = inertia_weight_max - (inertia_weight_max - inertia_weight_min) * progress_ratio

            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                chaotic_factor = self.chaotic_sequence[self.fitness_function_calls % len(self.chaotic_sequence)]
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i]) * chaotic_factor
                self.velocities[i] += social_constant * r2 * (self.global_best_position - self.particles[i]) * chaotic_factor
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Adaptive learning for diversification
            if progress_ratio > 0.5:  # Introduce additional perturbation in later stages
                perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                self.particles += perturbation * (1 - progress_ratio)

            # Improved elitist strategy to preserve better solutions
            elite_idxs = np.random.choice(np.argsort(self.personal_best_scores)[:max(1, self.population_size // 10)], max(1, self.population_size // 20), replace=False)
            elite_positions = self.personal_best_positions[elite_idxs]
            for i in range(len(elite_idxs)):
                self.particles[elite_idxs[i]] = elite_positions[i]

        return self.global_best_position