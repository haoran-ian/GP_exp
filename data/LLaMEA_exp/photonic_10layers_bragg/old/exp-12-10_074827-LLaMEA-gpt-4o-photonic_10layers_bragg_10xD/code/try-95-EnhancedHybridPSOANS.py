import numpy as np

class EnhancedHybridPSOANS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 2.2
        self.social_coefficient = 1.7
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.eval_count = 0
        self.inertia_min = 0.4
        self.inertia_max = 0.9

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
            dynamic_social_coefficient = self.social_coefficient * (1 - self.eval_count / self.budget)
            dynamic_cognitive_coefficient = self.cognitive_coefficient * (0.5 + 0.5 * (1 - self.eval_count / self.budget))
            cognitive_velocity = dynamic_cognitive_coefficient * r1 * (personal_best - particle)
            social_velocity = dynamic_social_coefficient * r2 * (global_best - particle)
            adaptive_scale = 1 / (1 + np.exp(-self.eval_count / (0.2 * self.budget)))
            new_velocity = (self.inertia_weight * velocity + cognitive_velocity + social_velocity) * adaptive_scale
            return new_velocity

        def fitness_driven_mutation(particle, global_best, value_diff):
            mutation_strength = 0.1 * (bounds[:, 1] - bounds[:, 0]) * (1 - value_diff)
            gradient_direction = global_best - particle
            mutation = np.random.uniform(-mutation_strength, mutation_strength, self.dim)
            gradient_influence = 0.5 * gradient_direction
            return mutation + gradient_influence

        while self.eval_count < self.budget:
            diversity = np.std(particles, axis=0).mean()
            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (self.eval_count / self.budget)) * (1 + diversity)

            for i in range(self.population_size):
                velocities[i] = update_particle_velocity(velocities[i], particles[i], personal_best_positions[i], global_best_position)
                particles[i] += velocities[i]

                value_diff = (personal_best_values[i] - global_best_value) / max(personal_best_values[i], global_best_value, 1e-9)
                if np.random.rand() < 0.15:
                    mutation = fitness_driven_mutation(particles[i], global_best_position, value_diff)
                    particles[i] += mutation

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
                perturbation_strength = np.std(np.abs(particles - global_best_position), axis=0).mean()
                perturbation = np.random.normal(0, perturbation_strength, self.dim)
                candidate = global_best_position + perturbation
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                candidate_value = func(candidate)
                self.eval_count += 1

                if candidate_value < global_best_value:
                    global_best_position = candidate
                    global_best_value = candidate_value

            self.temperature *= self.cooling_rate

        return global_best_position, global_best_value