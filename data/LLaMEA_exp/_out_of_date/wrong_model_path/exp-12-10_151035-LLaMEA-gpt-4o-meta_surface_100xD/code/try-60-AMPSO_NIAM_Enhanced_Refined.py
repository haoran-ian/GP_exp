import numpy as np

class AMPSO_NIAM_Enhanced_Refined:
    def __init__(self, budget, dim, population_size=30, w_initial=0.9, w_final=0.4, c1_initial=2.5, c2_initial=1.5, epsilon=0.1, mutation_chance=0.1, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w_initial = w_initial
        self.w_final = w_final
        self.c1_initial = c1_initial
        self.c2_initial = c2_initial
        self.epsilon = epsilon
        self.mutation_chance = mutation_chance
        self.global_best_position = None
        self.global_best_value = np.inf
        self.neighborhood_size = neighborhood_size

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

            # Calculate inertia weight and cognitive/social coefficients adaptively
            w = self.w_final + (self.w_initial - self.w_final) * np.exp(-3 * evaluations / self.budget)
            c1 = self.c1_initial - (self.c1_initial - 1.5) * (evaluations / self.budget)
            c2 = self.c2_initial + (2.5 - self.c2_initial) * (evaluations / self.budget)

            # Update particles' velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                # Adjust neighborhood size based on diversity
                adjusted_neighborhood_size = int(self.neighborhood_size + (np.std(personal_best_values) > self.epsilon) * 2)
                neighbors_indices = np.random.choice(self.population_size, adjusted_neighborhood_size, replace=False)
                local_best_index = neighbors_indices[np.argmin(personal_best_values[neighbors_indices])]
                local_best_position = personal_best_positions[local_best_index]

                cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = c2 * r2 * (local_best_position - particles[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component

                # Update particle position
                particles[i] += velocities[i]

                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)

                # Apply mutation to maintain diversity adaptively
                adaptive_mutation_chance = self.mutation_chance + (np.std(personal_best_values) < self.epsilon) * 0.1
                if np.random.rand() < adaptive_mutation_chance:
                    best_improvement = (personal_best_values[i] - value) / (personal_best_values[i] + 1e-10)
                    mutation_vector_scale = 0.1 * (1 - evaluations / self.budget) * (1 + best_improvement)
                    mutation_vector = np.random.normal(0, 1, self.dim) * (upper_bounds - lower_bounds) * mutation_vector_scale
                    particles[i] = np.clip(particles[i] + mutation_vector, lower_bounds, upper_bounds)

            # Maintain diversity by reinitializing stagnant particles
            if np.std(personal_best_values) < self.epsilon:
                stagnant_indices = np.where(personal_best_values > np.percentile(personal_best_values, 75))[0]
                for idx in stagnant_indices:
                    particles[idx] = np.random.uniform(lower_bounds, upper_bounds, self.dim)
                    velocities[idx] = np.random.uniform(-1, 1, self.dim)  # Reinitialize velocities
                    personal_best_positions[idx] = particles[idx].copy()
                    personal_best_values[idx] = np.inf

        return self.global_best_position, self.global_best_value