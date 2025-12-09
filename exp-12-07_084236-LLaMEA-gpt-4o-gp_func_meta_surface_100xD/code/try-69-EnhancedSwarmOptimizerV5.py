import numpy as np

class EnhancedSwarmOptimizerV5:
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
        stagnation_threshold = 10
        no_progress_count = 0
        last_best_score = np.inf
        chaos_factor = np.random.uniform(0, 1, self.dim)

        def levy_flight(Lambda=1.5):
            sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
                     (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
            u = np.random.normal(0, sigma, size=self.dim)
            v = np.random.normal(0, 1, size=self.dim)
            step = u / abs(v) ** (1 / Lambda)
            return step

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
            inertia_weight = 0.4 + chaos_factor * 0.2 * (1 - progress_ratio)

            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i])
                self.velocities[i] += social_constant * r2 * (self.global_best_position - self.particles[i])
                levy_step = levy_flight()
                self.velocities[i] += levy_step
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if self.global_best_score < last_best_score:
                last_best_score = self.global_best_score
                no_progress_count = 0
            else:
                no_progress_count += 1

            if no_progress_count >= stagnation_threshold:
                reinit_indices = np.random.choice(self.population_size, size=self.population_size // 4, replace=False)
                self.particles[reinit_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(reinit_indices), self.dim))
                no_progress_count = 0

            elite_idxs = np.random.choice(np.argsort(self.personal_best_scores)[:max(1, self.population_size // 10)], max(1, self.population_size // 20), replace=False)
            elite_positions = self.personal_best_positions[elite_idxs]
            for i in range(len(elite_idxs)):
                self.particles[elite_idxs[i]] = elite_positions[i]

        return self.global_best_position