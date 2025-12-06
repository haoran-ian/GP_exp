import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30  # Number of particles
        self.velocity_clamp = (-1.0, 1.0)
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 1.7  # Cognitive (personal) weight
        self.c2_max = 1.5  # Max social (global) weight
        self.cooling_rate = 0.99  # Cooling rate for SA
        self.temp = 1.0  # Initial temperature for SA
        self.successful_steps = 0  # Track successful steps

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(self.velocity_clamp[0], self.velocity_clamp[1], (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        while evaluations < self.budget:
            self.w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            self.c2 = self.c2_max * (1 - evaluations / self.budget)
            self.c1 = self.c1 * (1 - evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - particles[i])
                                 + self.c2 * r2 * (global_best_position - particles[i]))
                velocities[i] = np.clip(velocities[i], self.velocity_clamp[0], self.velocity_clamp[1])

                # Introduce adaptive velocity scaling
                scale_factor = 1 + (0.1 * self.successful_steps / self.population_size)
                particles[i] += scale_factor * velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                score = func(particles[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = np.copy(particles[i])
                    self.successful_steps += 1  # Increment successful steps

                if score < global_best_score or np.exp((global_best_score - score) / self.temp) > np.random.rand():
                    global_best_score = score
                    global_best_position = np.copy(particles[i])

            # Enhance local search around global best
            local_search = global_best_position + np.random.normal(0, 0.1, self.dim)
            local_search = np.clip(local_search, self.lower_bound, self.upper_bound)
            local_score = func(local_search)
            if local_score < global_best_score:
                global_best_score = local_score
                global_best_position = local_search

            self.temp *= self.cooling_rate

        return global_best_position, global_best_score