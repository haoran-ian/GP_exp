import numpy as np

class PSO_GA_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.8  # Modified inertia weight
        self.cognitive_coeff = 1.4  # Modified cognitive coefficient
        self.social_coeff = 1.4  # Modified social coefficient
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.mutation_rate = 0.1
        self.current_evals = 0
        self.f = 0.5  # New DE scale factor
        self.cr = 0.9  # New DE crossover probability

        # Initialize particles
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')

    def __call__(self, func):
        while self.current_evals < self.budget:
            for i in range(self.population_size):
                score = func(self.particles[i])
                self.current_evals += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = self.social_coeff * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.particles[i] += self.velocities[i]

                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    self.particles[i] += mutation_vector
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.particles[idxs]
                mutant_vector = x1 + self.f * (x2 - x3)
                trial_vector = np.copy(self.particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.cr:
                        trial_vector[j] = mutant_vector[j]
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                trial_score = func(trial_vector)
                self.current_evals += 1
                if trial_score < self.personal_best_scores[i] or \
                   np.exp((self.personal_best_scores[i] - trial_score) / self.temperature) > np.random.rand():
                    self.particles[i] = trial_vector
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector

            self.temperature *= self.cooling_rate

        return self.global_best_position, self.global_best_score