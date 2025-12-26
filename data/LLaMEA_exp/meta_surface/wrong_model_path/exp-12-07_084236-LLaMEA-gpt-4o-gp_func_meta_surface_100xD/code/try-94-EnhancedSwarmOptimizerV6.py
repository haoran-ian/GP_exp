import numpy as np

class EnhancedSwarmOptimizerV6:
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
        self.stagnation_threshold = 10
        self.no_progress_count = 0
        self.last_best_score = np.inf
        self.chaos_sequence = self.init_chaotic_sequence()

    def init_chaotic_sequence(self):
        x = 0.7  # Initial value for chaotic map
        chaos_seq = []
        for _ in range(self.population_size):
            x = 4 * x * (1 - x)  # Logistic map
            chaos_seq.append(x)
        return np.array(chaos_seq)

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1/beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1/beta)
        return step

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
            inertia_weight = 0.4 * (1 - progress_ratio**2)  # Modified line for adaptive inertia weight

            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                chaotic_factor = self.chaos_sequence[i]
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i])
                self.velocities[i] += social_constant * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] *= chaotic_factor  # Incorporate chaotic influence
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if self.global_best_score < self.last_best_score:
                self.last_best_score = self.global_best_score
                self.no_progress_count = 0
            else:
                self.no_progress_count += 1

            if self.no_progress_count >= self.stagnation_threshold:
                # Apply Levy flight for global exploration
                levy_indices = np.random.choice(self.population_size, size=self.population_size // 4, replace=False)
                steps = self.levy_flight((len(levy_indices), self.dim))
                self.particles[levy_indices] += steps
                self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)
                self.no_progress_count = 0

            # Improved elitist strategy to preserve better solutions
            elite_idxs = np.random.choice(np.argsort(self.personal_best_scores)[:max(1, self.population_size // 10)], max(1, self.population_size // 20), replace=False)
            elite_positions = self.personal_best_positions[elite_idxs]
            for i in range(len(elite_idxs)):
                self.particles[elite_idxs[i]] = elite_positions[i]

        return self.global_best_position