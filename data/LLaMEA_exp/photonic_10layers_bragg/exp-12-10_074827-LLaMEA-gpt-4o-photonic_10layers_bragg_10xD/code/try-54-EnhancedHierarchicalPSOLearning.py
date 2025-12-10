import numpy as np

class EnhancedHierarchicalPSOLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 1.5
        self.learning_step = 0.1
        self.adaptive_exploration_rate = 0.25
        self.eval_count = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx, :]
        global_best_value = personal_best_values[global_best_idx]

        def update_particle_velocity(velocity, particle, personal_best, global_best):
            r1, r2 = np.random.rand(2)
            cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best - particle)
            social_velocity = self.social_coefficient * r2 * (global_best - particle)
            new_velocity = self.inertia_weight * velocity + cognitive_velocity + social_velocity
            return new_velocity

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                velocities[i] = update_particle_velocity(velocities[i], particles[i], personal_best_positions[i], global_best_position)
                particles[i] += velocities[i]
                exploration_factor = np.exp(-self.eval_count / (0.5 * self.budget))
                
                if np.random.rand() < exploration_factor * self.adaptive_exploration_rate:
                    learning_factor = np.random.uniform(-self.learning_step, self.learning_step, self.dim)
                    particles[i] += learning_factor * (particles[i] - global_best_position)
                    
                particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
                current_value = func(particles[i])
                self.eval_count += 1

                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_values[i] = current_value

                    if current_value < global_best_value:
                        global_best_position = particles[i]
                        global_best_value = current_value

            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (self.eval_count / self.budget))

        return global_best_position, global_best_value