import numpy as np

class EnhancedHybridPSOANS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight_exploration = 0.9
        self.inertia_weight_exploitation = 0.4
        self.cognitive_coefficient = 2.2
        self.social_coefficient = 1.7
        self.temperature = 1.0
        self.cooling_rate = 0.95
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
            inertia_weight_dynamic = self.inertia_weight_exploration + (self.eval_count / self.budget) * (self.inertia_weight_exploitation - self.inertia_weight_exploration)
            cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best - particle)
            social_velocity = self.social_coefficient * r2 * (global_best - particle)
            new_velocity = inertia_weight_dynamic * velocity + cognitive_velocity + social_velocity
            return new_velocity

        def density_aware_mutation(particle, global_best):
            mutation_strength = 0.2 * (bounds[:, 1] - bounds[:, 0])
            density_factor = np.mean(np.linalg.norm(particles - particle, axis=1))
            mutation = np.random.uniform(-mutation_strength, mutation_strength, self.dim) * (1 / (1 + density_factor))
            return mutation

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                velocities[i] = update_particle_velocity(velocities[i], particles[i], personal_best_positions[i], global_best_position)
                particles[i] += velocities[i]

                if np.random.rand() < 0.15:
                    mutation = density_aware_mutation(particles[i], global_best_position)
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