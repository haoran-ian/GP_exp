import numpy as np

class PSO_GA_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 20
        self.max_population_size = 40
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.mutation_rate = 0.1
        self.current_evals = 0

        # Initialize particles
        self.population_size = self.initial_population_size
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')

    def __call__(self, func):
        while self.current_evals < self.budget:
            # Dynamic adjustments
            dynamic_inertia_weight = 0.4 + 0.5 * (1 - self.current_evals / self.budget)
            adjusted_cooling_rate = 0.99 + 0.01 * (self.current_evals / self.budget)
            dynamic_population_size = int(self.initial_population_size + 
                                          (self.max_population_size - self.initial_population_size) * 
                                          (self.current_evals / self.budget))
            velocity_scaling_factor = 1 + 0.5 * (self.current_evals / self.budget)

            # Adjust population size
            if dynamic_population_size > self.population_size:
                additional_particles = dynamic_population_size - self.population_size
                new_particles = np.random.uniform(self.lower_bound, self.upper_bound, (additional_particles, self.dim))
                self.particles = np.vstack((self.particles, new_particles))
                new_velocities = np.random.uniform(-1, 1, (additional_particles, self.dim))
                self.velocities = np.vstack((self.velocities, new_velocities))
                new_personal_best_scores = np.full(additional_particles, float('inf'))
                self.personal_best_scores = np.concatenate((self.personal_best_scores, new_personal_best_scores))
                self.personal_best_positions = np.vstack((self.personal_best_positions, new_particles))
                self.population_size = dynamic_population_size

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
                self.velocities[i] = (dynamic_inertia_weight * self.velocities[i] + 
                                      cognitive_velocity + social_velocity) * velocity_scaling_factor
                self.particles[i] += self.velocities[i]

                # Ensure particles are within bounds
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                
                # Apply mutation with dynamic rate
                dynamic_mutation_rate = self.mutation_rate * (1 - self.current_evals / self.budget)
                if np.random.rand() < dynamic_mutation_rate:
                    mutation_vector = np.random.normal(0, 1, self.dim)
                    self.particles[i] += mutation_vector
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Simulated Annealing-like exploration
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                candidate_solution = self.particles[i] + np.random.normal(0, 1, self.dim)
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
            self.temperature *= adjusted_cooling_rate

        return self.global_best_position, self.global_best_score