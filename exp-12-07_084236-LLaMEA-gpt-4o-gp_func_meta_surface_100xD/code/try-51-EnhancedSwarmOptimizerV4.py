import numpy as np

class EnhancedSwarmOptimizerV4:
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
        self.memory = np.random.rand(self.population_size, dim)

    def __call__(self, func):
        c1_initial, c2_initial = 2.5, 2.5
        c1_final, c2_final = 0.5, 0.5
        stagnation_threshold = 15
        no_progress_count = 0
        last_best_score = np.inf
        
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

            # Adaptive memory for learning coefficients
            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio + np.mean(self.memory) * 0.1
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio + np.std(self.memory) * 0.1

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i])
                self.velocities[i] += social_constant * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if self.global_best_score < last_best_score:
                last_best_score = self.global_best_score
                no_progress_count = 0
            else:
                no_progress_count += 1

            if no_progress_count >= stagnation_threshold:
                # Reinitialize some particles to improve diversity
                reinit_indices = np.random.choice(self.population_size, size=self.population_size // 5, replace=False)
                self.particles[reinit_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(reinit_indices), self.dim))
                no_progress_count = 0

            # Hybrid local search: Gaussian perturbation on top performers
            elite_idxs = np.argsort(self.personal_best_scores)[:max(1, self.population_size // 10)]
            for idx in elite_idxs:
                perturbed = self.personal_best_positions[idx] + np.random.normal(0, 0.1, self.dim)
                perturbed = np.clip(perturbed, self.lower_bound, self.upper_bound)
                if self.fitness_function_calls < self.budget:
                    perturbed_fitness = func(perturbed)
                    self.fitness_function_calls += 1
                    if perturbed_fitness < self.personal_best_scores[idx]:
                        self.personal_best_scores[idx] = perturbed_fitness
                        self.personal_best_positions[idx] = perturbed

        return self.global_best_position