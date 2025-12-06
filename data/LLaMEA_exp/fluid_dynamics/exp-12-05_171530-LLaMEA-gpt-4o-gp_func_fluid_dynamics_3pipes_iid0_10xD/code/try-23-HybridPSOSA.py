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

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(self.velocity_clamp[0], self.velocity_clamp[1], (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        while evaluations < self.budget:
            # Non-linear decrease inertia weight
            self.w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget)**2)
            self.c2 = self.c2_max * (1 - evaluations / self.budget) # Dynamic adjustment of social weight
            self.c1 = self.c1 * (1 - evaluations / self.budget)  # Adaptive personal weight adjustment

            for i in range(self.population_size):
                # PSO velocity update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - particles[i])
                                 + self.c2 * r2 * (global_best_position - particles[i]))
                
                # Adaptive velocity clamp
                velocities[i] = np.clip(velocities[i], 
                                        self.velocity_clamp[0] * (1 - evaluations / self.budget), 
                                        self.velocity_clamp[1] * (1 - evaluations / self.budget))

                # PSO position update
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                score = func(particles[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = np.copy(particles[i])

                # Simulated annealing acceptance criterion
                if score < global_best_score or np.exp((global_best_score - score) / self.temp) > np.random.rand():
                    # Update global best
                    global_best_score = score
                    global_best_position = np.copy(particles[i])

            # Cooling down the temperature
            self.temp *= self.cooling_rate

        return global_best_position, global_best_score