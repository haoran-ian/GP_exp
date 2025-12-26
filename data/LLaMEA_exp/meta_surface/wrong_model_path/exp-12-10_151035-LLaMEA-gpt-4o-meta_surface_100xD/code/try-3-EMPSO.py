import numpy as np

class EMPSO:
    def __init__(self, budget, dim, population_size=30, w_min=0.2, w_max=0.9, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w_min = w_min  # minimum inertia weight
        self.w_max = w_max  # maximum inertia weight
        self.c1 = c1  # cognitive (personal) coefficient
        self.c2 = c2  # social (global) coefficient
        self.global_best_position = None
        self.global_best_value = np.inf

    def __call__(self, func):
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initialize particles
        particles = np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Evaluate current position
                value = func(particles[i])
                evaluations += 1

                # Update personal best
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = particles[i].copy()

                # Update global best
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = particles[i].copy()

            # Calculate inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

            # Update particles' velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - particles[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component

                # Velocity clamping
                velocities[i] = np.clip(velocities[i], lower_bounds - upper_bounds, upper_bounds - lower_bounds)

                # Update particle position
                particles[i] += velocities[i]

                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)

        return self.global_best_position, self.global_best_value