import numpy as np

class Enhanced_PSO_GA_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.mutation_rate = 0.1
        self.current_evals = 0

        # Initialize particles
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')

    def levy_flight(self, size, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                    (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / np.abs(v) ** (1 / beta)
        return 0.1 * step

    def __call__(self, func):
        while self.current_evals < self.budget:
            for i in range(self.population_size):
                # Evaluate fitness
                score = func(self.particles[i])
                self.current_evals += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()

                # Update velocity and position
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = self.social_coeff * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.particles[i] += self.velocities[i]

                # Ensure particles are within bounds
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

                # Adaptive mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_strength = np.random.rand() * (self.upper_bound - self.lower_bound) * 0.01
                    mutation_vector = np.random.normal(0, mutation_strength, self.dim)
                    self.particles[i] += mutation_vector
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Simulated Annealing-like exploration with Lévy flight
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                candidate_solution = self.particles[i] + self.levy_flight(self.dim)
                candidate_solution = np.clip(candidate_solution, self.lower_bound, self.upper_bound)
                candidate_score = func(candidate_solution)
                self.current_evals += 1
                if candidate_score < self.personal_best_scores[i] or \
                   np.exp((self.personal_best_scores[i] - candidate_score) / self.temperature) > np.random.rand():
                    self.particles[i] = candidate_solution
                    self.personal_best_scores[i] = candidate_score
                    if candidate_score < self.global_best_score:
                        self.global_best_score = candidate_score
                        self.global_best_position = candidate_solution

            # Cool down the temperature
            self.temperature *= self.cooling_rate

        return self.global_best_position, self.global_best_score