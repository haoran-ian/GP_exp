import numpy as np

class AMPSO_NIAM:
    def __init__(self, budget, dim, population_size=30, w_max=0.9, w_min=0.4, c1=2.05, c2=2.05, epsilon=0.1, mutation_chance=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w_max = w_max  # maximum inertia weight
        self.w_min = w_min  # minimum inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.global_best_position = None
        self.global_best_value = np.inf
        self.epsilon = epsilon  # diversity threshold
        self.mutation_chance = mutation_chance  # chance of mutation

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

            # Calculate inertia weight using nonlinear function
            w = self.w_min + (self.w_max - self.w_min) * (1 - (evaluations / self.budget)**2)

            # Update particles' velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - particles[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component

                # Update particle position
                particles[i] += velocities[i]

                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)

                # Apply mutation to maintain diversity
                if np.random.rand() < self.mutation_chance:
                    mutation_vector = np.random.normal(0, 1, self.dim) * (upper_bounds - lower_bounds) * 0.1
                    particles[i] = np.clip(particles[i] + mutation_vector, lower_bounds, upper_bounds)

            # Maintain diversity by reinitializing stagnant particles
            if np.std(personal_best_values) < self.epsilon:
                stagnant_indices = np.where(personal_best_values > np.percentile(personal_best_values, 75))[0]
                for idx in stagnant_indices:
                    particles[idx] = np.random.uniform(lower_bounds, upper_bounds, self.dim)
                    personal_best_positions[idx] = particles[idx].copy()
                    personal_best_values[idx] = np.inf

        return self.global_best_position, self.global_best_value