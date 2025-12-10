import numpy as np

class EnhancedSwarmOptimizerV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, max(10, dim * 2))
        self.particles = self._initialize_particles_chaotically()
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_function_calls = 0
        self.velocity_clamp = 0.1 * (self.upper_bound - self.lower_bound)  # Adaptive velocity clamping

    def _initialize_particles_chaotically(self):
        x = np.linspace(0, 1, self.population_size)
        logistic_map = 4.0 * x * (1 - x)  # Chaotic sequence
        return self.lower_bound + logistic_map[:, None] * (self.upper_bound - self.lower_bound)

    def __call__(self, func):
        c1_initial, c2_initial = 2.5, 2.5
        c1_final, c2_final = 0.5, 0.5
        neighborhood_size = max(3, self.population_size // 5)

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
            inertia_weight = 0.4 + 0.2 * np.random.rand() * (1 - progress_ratio)

            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

            for i in range(self.population_size):
                neighbors_idxs = np.random.choice(self.population_size, neighborhood_size, replace=False)
                local_best_idx = neighbors_idxs[np.argmin(self.personal_best_scores[neighbors_idxs])]
                local_best_position = self.personal_best_positions[local_best_idx]

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i])
                self.velocities[i] += social_constant * r2 * (local_best_position - self.particles[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            opposition_particles = self.lower_bound + self.upper_bound - self.particles
            for i in range(self.population_size):
                if self.fitness_function_calls >= self.budget:
                    break

                opp_fitness = func(opposition_particles[i])
                self.fitness_function_calls += 1

                if opp_fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = opp_fitness
                    self.personal_best_positions[i] = opposition_particles[i]

                    if opp_fitness < self.global_best_score:
                        self.global_best_score = opp_fitness
                        self.global_best_position = opposition_particles[i]

            elite_idxs = np.random.choice(np.argsort(self.personal_best_scores)[:max(1, self.population_size // 10)], max(1, self.population_size // 20), replace=False)
            elite_positions = self.personal_best_positions[elite_idxs]
            for i in range(len(elite_idxs)):
                self.particles[elite_idxs[i]] = elite_positions[i]

        return self.global_best_position