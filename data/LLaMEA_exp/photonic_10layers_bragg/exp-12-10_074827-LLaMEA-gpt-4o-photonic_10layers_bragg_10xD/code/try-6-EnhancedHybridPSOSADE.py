import numpy as np

class EnhancedHybridPSOSADE:
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
        self.F = 0.8 # DE mutation factor
        self.CR = 0.9 # DE crossover probability

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
            new_velocity = (self.inertia_weight * velocity + cognitive_velocity + social_velocity)
            return new_velocity

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                # DE mutation and crossover
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = particles[idxs]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), bounds[:, 0], bounds[:, 1])
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant_vector, particles[i])
                trial_value = func(trial_vector)
                self.eval_count += 1

                if trial_value < personal_best_values[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_values[i] = trial_value

                    if trial_value < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = trial_value
                
                # Update velocities and positions
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

            # Simulated Annealing step
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