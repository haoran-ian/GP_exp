import numpy as np

class EnhancedHybridPSOANSV2:
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
        self.memory_pool = []  # Memory pool to hold best particles for diversification
        self.memory_size = 5

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
            new_velocity = self.inertia_weight * velocity + cognitive_velocity + social_velocity
            return new_velocity

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
                        self.memory_pool.append(global_best_position)
                        if len(self.memory_pool) > self.memory_size:
                            self.memory_pool.pop(0)

            if np.random.rand() < 0.1:  # Probability of reinitializing velocities
                for i in range(self.population_size):
                    velocities[i] = np.random.uniform(-1, 1, self.dim)

            if len(self.memory_pool) > 0 and np.random.rand() < 0.2:
                candidate = self.memory_pool[np.random.choice(len(self.memory_pool))]
                candidate_value = func(candidate)
                self.eval_count += 1

                if candidate_value < global_best_value:
                    global_best_position = candidate
                    global_best_value = candidate_value

        return global_best_position, global_best_value