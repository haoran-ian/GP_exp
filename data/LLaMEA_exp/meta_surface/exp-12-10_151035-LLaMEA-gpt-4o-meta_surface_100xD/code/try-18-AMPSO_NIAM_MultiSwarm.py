import numpy as np

class AMPSO_NIAM_MultiSwarm:
    def __init__(self, budget, dim, population_size=30, w_initial=0.9, w_final=0.4, c1_initial=2.5, c2_initial=1.5, epsilon=0.1, mutation_chance=0.1, num_swarms=3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.w_initial = w_initial
        self.w_final = w_final
        self.c1_initial = c1_initial
        self.c2_initial = c2_initial
        self.epsilon = epsilon
        self.mutation_chance = mutation_chance
        self.num_swarms = num_swarms
        self.global_best_position = None
        self.global_best_value = np.inf

    def __call__(self, func):
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Split population into multiple swarms
        swarm_size = self.population_size // self.num_swarms
        swarms = [np.random.uniform(lower_bounds, upper_bounds, (swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (swarm_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_values = [np.full(swarm_size, np.inf) for _ in range(self.num_swarms)]

        evaluations = 0

        while evaluations < self.budget:
            for swarm_idx in range(self.num_swarms):
                for i in range(swarm_size):
                    if evaluations >= self.budget:
                        break

                    # Evaluate current position
                    value = func(swarms[swarm_idx][i])
                    evaluations += 1

                    # Update personal best
                    if value < personal_best_values[swarm_idx][i]:
                        personal_best_values[swarm_idx][i] = value
                        personal_best_positions[swarm_idx][i] = swarms[swarm_idx][i].copy()

                    # Update global best
                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best_position = swarms[swarm_idx][i].copy()

            # Calculate inertia weight and cognitive/social coefficients adaptively
            w = self.w_final + (self.w_initial - self.w_final) * np.exp(-3 * evaluations / self.budget)
            c1 = self.c1_initial - (self.c1_initial - 1.5) * (evaluations / self.budget)
            c2 = self.c2_initial + (2.5 - self.c2_initial) * (evaluations / self.budget)

            # Update particles' velocities and positions
            for swarm_idx in range(self.num_swarms):
                for i in range(swarm_size):
                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive_component = c1 * r1 * (personal_best_positions[swarm_idx][i] - swarms[swarm_idx][i])
                    social_component = c2 * r2 * (self.global_best_position - swarms[swarm_idx][i])
                    velocities[swarm_idx][i] = w * velocities[swarm_idx][i] + cognitive_component + social_component

                    # Update particle position
                    swarms[swarm_idx][i] += velocities[swarm_idx][i]

                    # Ensure particles are within bounds
                    swarms[swarm_idx][i] = np.clip(swarms[swarm_idx][i], lower_bounds, upper_bounds)

                    # Apply mutation to maintain diversity adaptively
                    adaptive_mutation_chance = self.mutation_chance + (np.std(personal_best_values[swarm_idx]) < self.epsilon) * 0.1
                    if np.random.rand() < adaptive_mutation_chance:
                        mutation_vector = np.random.normal(0, 1, self.dim) * (upper_bounds - lower_bounds) * 0.1
                        swarms[swarm_idx][i] = np.clip(swarms[swarm_idx][i] + mutation_vector, lower_bounds, upper_bounds)

                # Maintain diversity by reinitializing stagnant particles
                if np.std(personal_best_values[swarm_idx]) < self.epsilon:
                    stagnant_indices = np.where(personal_best_values[swarm_idx] > np.percentile(personal_best_values[swarm_idx], 75))[0]
                    for idx in stagnant_indices:
                        swarms[swarm_idx][idx] = np.random.uniform(lower_bounds, upper_bounds, self.dim)
                        personal_best_positions[swarm_idx][idx] = swarms[swarm_idx][idx].copy()
                        personal_best_values[swarm_idx][idx] = np.inf

        return self.global_best_position, self.global_best_value