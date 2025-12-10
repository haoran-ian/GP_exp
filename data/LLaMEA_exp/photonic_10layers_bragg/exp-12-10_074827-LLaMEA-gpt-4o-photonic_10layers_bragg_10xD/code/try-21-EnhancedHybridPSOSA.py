import numpy as np

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.99
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

        def update_particle_velocity(velocity, particle, personal_best, global_best, eval_progress):
            r1, r2 = np.random.rand(2)
            adaptive_cognitive = self.cognitive_coefficient * (1 - eval_progress)
            adaptive_social = self.social_coefficient * eval_progress
            cognitive_velocity = adaptive_cognitive * r1 * (personal_best - particle)
            social_velocity = adaptive_social * r2 * (global_best - particle)
            random_scaling = np.random.uniform(0.3, 1.7)
            new_velocity = (self.inertia_weight * velocity + cognitive_velocity + social_velocity) * random_scaling
            return new_velocity

        while self.eval_count < self.budget:
            eval_progress = self.eval_count / self.budget
            for i in range(self.population_size):
                self.inertia_weight = 0.9 - 0.8 * eval_progress

                velocities[i] = update_particle_velocity(velocities[i], particles[i], personal_best_positions[i], global_best_position, eval_progress)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
                current_value = func(particles[i])
                self.eval_count += 1

                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_values[i] = current_value

                    if current_value < global_best_value:
                        global_best_position = particles[i]
                        global_best_value = current_value

            if np.random.rand() < self.temperature:
                perturbation = np.random.uniform(-1, 1, self.dim)
                candidate = global_best_position + perturbation
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                candidate_value = func(candidate)
                self.eval_count += 1

                if candidate_value < global_best_value or np.random.rand() < np.exp((global_best_value - candidate_value) / self.temperature):
                    global_best_position = candidate
                    global_best_value = candidate_value

            self.temperature *= self.cooling_rate

        return global_best_position, global_best_value