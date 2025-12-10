import numpy as np

class Quantum_Enhanced_Adaptive_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 25
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.mutation_rate = 0.15
        self.current_evals = 0

        # Initialize particles
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.elite_positions = np.copy(self.particles[:3])
        self.q_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def __call__(self, func):
        while self.current_evals < self.budget:
            dynamic_inertia_weight = 0.4 + 0.5 * (1 - self.current_evals / self.budget)
            for i in range(self.population_size):
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
                diversity_control = 0.1 * r3 * (self.elite_positions[np.random.randint(0, 3)] - self.particles[i])
                
                # Quantum-inspired update
                q_factor = np.random.normal(0, 1, self.dim)
                quantum_velocity = q_factor * (self.q_positions[i] - self.particles[i])

                self.velocities[i] = (dynamic_inertia_weight * self.velocities[i] +
                                      0.7 * cognitive_velocity + 0.7 * social_velocity + 0.7 * diversity_control + 0.05 * quantum_velocity)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

                # Local stochastic search
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    self.particles[i] += mutation_vector
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            sorted_indices = np.argsort(self.personal_best_scores)
            self.elite_positions = self.personal_best_positions[sorted_indices[:3]]

            # Update quantum positions based on new elite positions
            self.q_positions = np.copy(self.elite_positions)

            self.temperature *= self.cooling_rate

            self.cognitive_coeff = 1.5 + 0.2 * np.cos(2 * np.pi * self.current_evals / self.budget)
            self.social_coeff = 1.4 + 0.2 * np.cos(2 * np.pi * self.current_evals / self.budget)

        return self.global_best_position, self.global_best_score