import numpy as np

class Refined_Adaptive_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 20
        self.dynamic_population_size = self.initial_population_size
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.mutation_rate = 0.15
        self.current_evals = 0

        # Initialize particles
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.initial_population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.initial_population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')

    def __call__(self, func):
        while self.current_evals < self.budget:
            dynamic_inertia_weight = 0.4 + 0.5 * (1 - self.current_evals / self.budget)
            adaptive_cooling_rate = self.cooling_rate + 0.02 * np.sin(3 * np.pi * self.current_evals / self.budget)
            exploration_balance = 0.1 + 0.9 * (self.current_evals / self.budget)

            if self.current_evals % (self.budget // 5) == 0 and self.dynamic_population_size > 5:
                self.dynamic_population_size = max(5, int(self.dynamic_population_size * 0.9))
                sorted_indices = np.argsort(self.personal_best_scores)
                self.particles = self.personal_best_positions[sorted_indices[:self.dynamic_population_size]]
                self.velocities = self.velocities[sorted_indices[:self.dynamic_population_size]]
                self.personal_best_positions = self.personal_best_positions[sorted_indices[:self.dynamic_population_size]]
                self.personal_best_scores = self.personal_best_scores[sorted_indices[:self.dynamic_population_size]]

            for i in range(self.dynamic_population_size):
                score = func(self.particles[i])
                self.current_evals += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()

                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = self.social_coeff * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = (dynamic_inertia_weight * self.velocities[i] +
                                      0.8 * cognitive_velocity + 0.8 * social_velocity)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

                dynamic_mutation_rate = self.mutation_rate * (1 - self.current_evals / self.budget)
                if np.random.rand() < dynamic_mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1 + 0.1 * (np.sin(np.pi * self.current_evals / self.budget)), self.dim)
                    self.particles[i] += mutation_vector
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            self.cognitive_coeff = 1.5 + 0.2 * np.sin(2 * np.pi * self.current_evals / self.budget)
            self.social_coeff = 1.4 + 0.2 * np.sin(2 * np.pi * self.current_evals / self.budget)

        return self.global_best_position, self.global_best_score