import numpy as np

class EnhancedHybridPSOANS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 2.2
        self.social_coefficient = 1.7
        self.eval_count = 0
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.convergence_pressure = 0.1

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_idx = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_idx, :]
        global_best_value = personal_best_values[global_best_idx]

        def update_particle_velocity(velocity, particle, personal_best, global_best, influence_factor):
            r1, r2 = np.random.rand(2)
            cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best - particle)
            social_velocity = self.social_coefficient * r2 * (global_best - particle) * influence_factor
            adaptive_scale = 1 / (1 + np.exp(-self.eval_count / (0.2 * self.budget)))
            new_velocity = (self.inertia_weight * velocity + cognitive_velocity + social_velocity) * adaptive_scale
            return new_velocity

        def adaptive_influence_diversity():
            diversity_factor = np.std(particles, axis=0).mean()
            return 1 + self.convergence_pressure * (1 - diversity_factor / (bounds[:,1] - bounds[:,0]).mean())

        def gradient_informed_mutation(particle, global_best):
            mutation_strength = 0.1 * (bounds[:, 1] - bounds[:, 0])
            gradient_direction = global_best - particle
            mutation = np.random.uniform(-mutation_strength, mutation_strength, self.dim)
            gradient_influence = 0.5 * gradient_direction
            return mutation + gradient_influence

        while self.eval_count < self.budget:
            influence_factor = adaptive_influence_diversity()
            diversity = np.std(particles, axis=0).mean()
            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (self.eval_count / self.budget)) * (1 + diversity)

            for i in range(self.population_size):
                velocities[i] = update_particle_velocity(velocities[i], particles[i], personal_best_positions[i], global_best_position, influence_factor)
                particles[i] += velocities[i]

                if np.random.rand() < 0.15:
                    mutation = gradient_informed_mutation(particles[i], global_best_position)
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

        return global_best_position, global_best_value