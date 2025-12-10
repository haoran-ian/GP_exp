import numpy as np

class AdvancedHybridPSODE:
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
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

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
            cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best - particle)
            social_velocity = dynamic_social_coefficient * r2 * (global_best - particle)
            random_scaling = np.random.uniform(0.3, 1.7)
            new_velocity = (self.inertia_weight * velocity + cognitive_velocity + social_velocity) * random_scaling
            return new_velocity

        def differential_evolution_mutation(target_idx):
            indices = [idx for idx in range(self.population_size) if idx != target_idx]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = particles[a] + self.mutation_factor * (particles[b] - particles[c])
            mutant_vector = np.clip(mutant_vector, bounds[:, 0], bounds[:, 1])
            return mutant_vector

        while self.eval_count < self.budget:
            diversity = np.std(particles, axis=0).mean()
            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (self.eval_count / self.budget)) * (1 + diversity)

            for i in range(self.population_size):
                velocities[i] = update_particle_velocity(velocities[i], particles[i], personal_best_positions[i], global_best_position)
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

                # Apply differential evolution mutation
                mutant_vector = differential_evolution_mutation(i)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, particles[i])
                trial_value = func(trial_vector)
                self.eval_count += 1

                if trial_value < personal_best_values[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_values[i] = trial_value

                    if trial_value < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = trial_value

            if np.random.rand() < self.temperature:
                perturbation_strength = np.std(particles - global_best_position, axis=0).mean()
                perturbation = np.random.normal(0, perturbation_strength, self.dim)
                candidate = global_best_position + perturbation
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                candidate_value = func(candidate)
                self.eval_count += 1

                if candidate_value < global_best_value or np.random.rand() < np.exp((global_best_value - candidate_value) / self.temperature):
                    global_best_position = candidate
                    global_best_value = candidate_value

            self.temperature *= self.cooling_rate

        return global_best_position, global_best_value