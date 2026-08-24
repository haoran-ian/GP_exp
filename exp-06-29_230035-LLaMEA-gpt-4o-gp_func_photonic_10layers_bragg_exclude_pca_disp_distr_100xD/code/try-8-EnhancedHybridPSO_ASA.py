import numpy as np

class EnhancedHybridPSO_ASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, 10 * dim)  # Increased particles for better search space coverage
        self.inertia = 0.9  # Improved inertia weight for stability
        self.cognitive = 1.5  # Cognitive component
        self.social = 1.5  # Social component
        self.initial_temperature = 100.0  # Initial temperature for adaptive simulated annealing
        self.cooling_rate = 0.95  # Slightly slower cooling for broader search
        self.minimum_temperature = 1e-3  # Minimum temperature to maintain some randomness

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        particles = np.random.uniform(bounds[0], bounds[1], (self.num_particles, self.dim))
        velocities = np.random.rand(self.num_particles, self.dim)
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        evaluations = self.num_particles
        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # Update velocity
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.social * np.random.rand(self.dim) * (global_best - particles[i])
                )
                # Update position
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

                # Evaluate particle
                fitness = func(particles[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = fitness

                # Adaptive simulated annealing: adaptive probabilistic acceptance
                acceptance_probability = np.exp((personal_best_values[i] - global_best_value) / max(temperature, self.minimum_temperature))
                if fitness < global_best_value or np.random.rand() < acceptance_probability:
                    global_best = personal_best[i]
                    global_best_value = personal_best_values[i]

            # Cooling the temperature adaptively
            temperature *= self.cooling_rate

        return global_best