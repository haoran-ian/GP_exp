import numpy as np

class EnhancedHybridPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.93
        self.eval_count = 0
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability

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
            adaptive_scale = 1 / (1 + np.exp(-self.eval_count / (0.2 * self.budget)))
            new_velocity = (self.inertia_weight * velocity + cognitive_velocity + social_velocity) * adaptive_scale
            return new_velocity

        def differential_evolution(p):
            idxs = [idx for idx in range(self.population_size) if idx != p]
            a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
            cross_points = np.random.rand(self.dim) < self.CR
            trial = np.where(cross_points, mutant, particles[p])
            trial_value = func(trial)
            return trial, trial_value

        while self.eval_count < self.budget:
            diversity = np.std(particles, axis=0).mean()
            self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min) * (self.eval_count / self.budget)) * (1 + diversity)

            for i in range(self.population_size):
                if np.random.rand() < 0.3:  # Introduce a probability to apply DE
                    trial, trial_value = differential_evolution(i)
                    if trial_value < personal_best_values[i]:
                        personal_best_positions[i] = trial
                        personal_best_values[i] = trial_value

                        if trial_value < global_best_value:
                            global_best_position = trial
                            global_best_value = trial_value
                else:
                    velocities[i] = update_particle_velocity(velocities[i], particles[i], personal_best_positions[i], global_best_position)
                    particles[i] += velocities[i]

                    if np.random.rand() < 0.1:
                        mutation_strength = 0.1 * (bounds[:, 1] - bounds[:, 0])
                        mutation = np.random.uniform(-mutation_strength, mutation_strength, self.dim)
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
                perturbation_strength = np.std(particles - global_best_position, axis=0).mean()
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