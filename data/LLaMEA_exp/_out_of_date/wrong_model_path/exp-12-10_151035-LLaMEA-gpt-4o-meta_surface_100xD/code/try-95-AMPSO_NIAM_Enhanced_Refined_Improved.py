import numpy as np

class AMPSO_NIAM_Enhanced_Refined_Improved:
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

        particles = np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            # Create dynamic subgroups for enhanced exploration
            subgroup_size = max(2, self.population_size // 4)
            indices = np.random.permutation(self.population_size)

            for i in range(0, self.population_size, subgroup_size):
                subgroup_indices = indices[i:i + subgroup_size]

                for j in subgroup_indices:
                    if evaluations >= self.budget:
                        break

                    value = func(particles[j])
                    evaluations += 1

                    if value < personal_best_values[j]:
                        personal_best_values[j] = value
                        personal_best_positions[j] = particles[j].copy()

                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best_position = particles[j].copy()

                # Adaptive coefficients based on subgroup performance
                subgroup_best_value = np.min(personal_best_values[subgroup_indices])
                subgroup_adaptation_factor = 1 + (subgroup_best_value - self.global_best_value) / (abs(self.global_best_value) + 1e-10)

                w = self.w_final + (self.w_initial - self.w_final) * np.exp(-3 * evaluations / self.budget)
                c1 = self.c1_initial * subgroup_adaptation_factor
                c2 = self.c2_initial * subgroup_adaptation_factor

                for j in subgroup_indices:
                    r1, r2 = np.random.rand(), np.random.rand()
                    adjusted_neighborhood_size = int(self.neighborhood_size + (np.std(personal_best_values) > self.epsilon) * 2)
                    neighbors_indices = np.random.choice(subgroup_indices, adjusted_neighborhood_size, replace=False)
                    local_best_index = neighbors_indices[np.argmin(personal_best_values[neighbors_indices])]
                    local_best_position = personal_best_positions[local_best_index]

                    cognitive_component = c1 * r1 * (personal_best_positions[j] - particles[j])
                    social_component = c2 * r2 * (local_best_position - particles[j])
                    velocities[j] = w * velocities[j] + cognitive_component + social_component

                    particles[j] += velocities[j]
                    particles[j] = np.clip(particles[j], lower_bounds, upper_bounds)

                    adaptive_mutation_chance = self.mutation_chance + (np.std(personal_best_values) < self.epsilon) * 0.1
                    if np.random.rand() < adaptive_mutation_chance:
                        perturbation = (1 - evaluations / self.budget) * (1.5 - np.std(personal_best_values))
                        mutation_vector = np.random.normal(0, 1, self.dim) * (upper_bounds - lower_bounds) * perturbation
                        particles[j] = np.clip(particles[j] + mutation_vector, lower_bounds, upper_bounds)

                if np.std(personal_best_values[subgroup_indices]) < self.epsilon:
                    stagnant_indices = subgroup_indices[personal_best_values[subgroup_indices] > np.percentile(personal_best_values[subgroup_indices], 75)]
                    for idx in stagnant_indices:
                        particles[idx] = np.random.uniform(lower_bounds, upper_bounds, self.dim)
                        personal_best_positions[idx] = particles[idx].copy()
                        personal_best_values[idx] = np.inf

        return self.global_best_position, self.global_best_value