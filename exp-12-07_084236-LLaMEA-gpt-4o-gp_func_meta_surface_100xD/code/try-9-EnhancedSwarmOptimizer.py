import numpy as np

class EnhancedSwarmOptimizer:
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
        self.boundary_shrink_rate = 0.9  # New parameter for dynamic boundary adjustment

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
            inertia_weight = 0.4 + 0.3 * np.random.rand() * (1 - progress_ratio)

            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

            learning_rate = 0.1 + 0.9 * progress_ratio  # Progressive learning rate

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i])
                self.velocities[i] += social_constant * r2 * (self.global_best_position - self.particles[i])
                self.particles[i] += learning_rate * self.velocities[i]  # Apply learning rate
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            self.lower_bound *= self.boundary_shrink_rate  # Adjust boundaries dynamically
            self.upper_bound *= self.boundary_shrink_rate

            for i in range(self.population_size):
                if self.fitness_function_calls >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutation_factor = 0.6 + 0.4 * (1 - progress_ratio)
                mutant_vector = self.particles[a] + mutation_factor * (self.particles[b] - self.particles[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                crossover_vector = np.where(np.random.rand(self.dim) < 0.9, mutant_vector, self.particles[i])

                crossover_fitness = func(crossover_vector)
                self.fitness_function_calls += 1

                if crossover_fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = crossover_fitness
                    self.particles[i] = crossover_vector

                    if crossover_fitness < self.global_best_score:
                        self.global_best_score = crossover_fitness
                        self.global_best_position = crossover_vector

        return self.global_best_position