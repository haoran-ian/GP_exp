import numpy as np

class AMPSO_NIAM_Enhanced_Refined:
    def __init__(self, budget, dim, population_size=30, w_initial=0.9, w_final=0.4, c1_initial=2.5, c2_initial=1.5, epsilon=0.1, mutation_chance=0.1, neighborhood_size=5, num_swarms=2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w_initial = w_initial
        self.w_final = w_final
        self.c1_initial = c1_initial
        self.c2_initial = c2_initial
        self.epsilon = epsilon
        self.mutation_chance = mutation_chance
        self.global_best_position = [None] * num_swarms
        self.global_best_value = [np.inf] * num_swarms
        self.neighborhood_size = neighborhood_size
        self.num_swarms = num_swarms

    def __call__(self, func):
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initialize particles
        particles = [np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(p) for p in particles]
        personal_best_values = [np.full(self.population_size, np.inf) for _ in range(self.num_swarms)]

        evaluations = 0

        while evaluations < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.population_size):
                    if evaluations >= self.budget:
                        break

                    # Evaluate current position
                    value = func(particles[swarm][i])
                    evaluations += 1

                    # Update personal best
                    if value < personal_best_values[swarm][i]:
                        personal_best_values[swarm][i] = value
                        personal_best_positions[swarm][i] = particles[swarm][i].copy()

                    # Update global best
                    if value < self.global_best_value[swarm]:
                        self.global_best_value[swarm] = value
                        self.global_best_position[swarm] = particles[swarm][i].copy()

            # Calculate inertia weight and cognitive/social coefficients adaptively
            w = self.w_final + (self.w_initial - self.w_final) * np.exp(-3 * evaluations / self.budget)
            c1 = self.c1_initial - (self.c1_initial - 1.5) * (evaluations / self.budget)
            c2 = self.c2_initial + (2.5 - self.c2_initial) * (evaluations / self.budget)

            # Update particles' velocities and positions
            for swarm in range(self.num_swarms):
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(), np.random.rand()
                    adjusted_neighborhood_size = int(self.neighborhood_size + (np.std(personal_best_values[swarm]) > self.epsilon) * 2)
                    neighbors_indices = np.random.choice(self.population_size, adjusted_neighborhood_size, replace=False)
                    local_best_index = neighbors_indices[np.argmin(personal_best_values[swarm][neighbors_indices])]
                    local_best_position = personal_best_positions[swarm][local_best_index]

                    cognitive_component = c1 * r1 * (personal_best_positions[swarm][i] - particles[swarm][i])
                    social_component = c2 * r2 * (local_best_position - particles[swarm][i])
                    velocities[swarm][i] = w * velocities[swarm][i] + cognitive_component + social_component

                    # Update particle position
                    particles[swarm][i] += velocities[swarm][i]

                    # Ensure particles are within bounds
                    particles[swarm][i] = np.clip(particles[swarm][i], lower_bounds, upper_bounds)

                    # Apply mutation to maintain diversity adaptively
                    adaptive_mutation_chance = self.mutation_chance + (np.std(personal_best_values[swarm]) < self.epsilon) * 0.1
                    if np.random.rand() < adaptive_mutation_chance:
                        mutation_vector_scale = 0.1 * (1 - evaluations / self.budget)
                        mutation_vector = np.random.normal(0, 1, self.dim) * (upper_bounds - lower_bounds) * mutation_vector_scale
                        particles[swarm][i] = np.clip(particles[swarm][i] + mutation_vector, lower_bounds, upper_bounds)

            # Inter-swarm cooperation to enhance exploration
            if self.num_swarms > 1:
                for swarm in range(self.num_swarms - 1):
                    if self.global_best_value[swarm] < self.global_best_value[swarm + 1]:
                        self.global_best_position[swarm + 1] = self.global_best_position[swarm]
                        self.global_best_value[swarm + 1] = self.global_best_value[swarm]

        final_best_swarm = np.argmin(self.global_best_value)
        return self.global_best_position[final_best_swarm], self.global_best_value[final_best_swarm]