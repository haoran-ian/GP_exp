import numpy as np

class EnhancedSwarmOptimizerV3:
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
        c1_initial, c2_initial = 2.0, 2.0  # Adjusted coefficients for better balance
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
            inertia_weight = 0.5 + 0.4 * np.random.rand() * (1 - progress_ratio)  # Adjusted inertia

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

            # Adaptive boundary control
            boundary_correction = np.where((self.particles < self.lower_bound) | (self.particles > self.upper_bound), 
                                           self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound), 
                                           self.particles)
            self.particles = np.where(np.random.rand(self.population_size, self.dim) < 0.1, boundary_correction, self.particles)

            # Hybrid updates inspired by DE
            for i in range(self.population_size):
                if self.fitness_function_calls >= self.budget:
                    break
                if np.random.rand() < 0.2:
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    mutant_vector = self.personal_best_positions[a] + 0.5 * (self.personal_best_positions[b] - self.personal_best_positions[c])
                    trial_vector = np.where(np.random.rand(self.dim) < 0.5, mutant_vector, self.particles[i])
                    trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                    trial_fitness = func(trial_vector)
                    self.fitness_function_calls += 1

                    if trial_fitness < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_fitness
                        self.personal_best_positions[i] = trial_vector

                        if trial_fitness < self.global_best_score:
                            self.global_best_score = trial_fitness
                            self.global_best_position = trial_vector

        return self.global_best_position