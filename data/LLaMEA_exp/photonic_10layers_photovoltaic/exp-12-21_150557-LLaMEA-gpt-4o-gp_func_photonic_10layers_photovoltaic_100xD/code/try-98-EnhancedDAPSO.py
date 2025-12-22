import numpy as np

class EnhancedDAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.c1_start, self.c1_end = 2.5, 0.5   # dynamic cognitive component
        self.c2_start, self.c2_end = 0.5, 2.5   # dynamic social component
        self.w_start, self.w_end = 0.9, 0.4     # dynamic inertia weight
        self.global_best_position = None
        self.global_best_value = float('inf')
    
    def chaotically_mutate(self, position, lb, ub, progress):
        beta = 0.6  # mutation intensity, increased from 0.5 to enhance exploration
        chaos_factor = 0.5 + 3.5 * progress * (1 - progress)  # updated logistic map with a dynamic base
        mutation = beta * chaos_factor * (np.random.rand(self.dim) - 0.5)
        return np.clip(position + mutation, lb, ub)
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particle_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        particle_velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = particle_positions.copy()
        personal_best_values = np.full(self.population_size, float('inf'))

        evaluations = 0

        while evaluations < self.budget:
            subgroup_size = max(2, int((np.sin((evaluations / self.budget) * np.pi) + 1) / 2 * self.population_size))  # dynamic subgroup size
            subgroup_indices = np.random.choice(self.population_size, subgroup_size, replace=False)
            subgroup_best_position, subgroup_best_value = None, float('inf')

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(particle_positions[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particle_positions[i]

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = particle_positions[i]

                if i in subgroup_indices and current_value < subgroup_best_value:
                    subgroup_best_value = current_value
                    subgroup_best_position = particle_positions[i]

            progress = evaluations / self.budget
            c1 = self.c1_start * (1 - progress) + self.c1_end * progress
            c2 = self.c2_start * (1 - progress) + self.c2_end * progress
            w = self.w_start * (1 - progress) + self.w_end * progress

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = c1 * r1 * (personal_best_positions[i] - particle_positions[i])
                social_velocity = c2 * r2 * (subgroup_best_position - particle_positions[i] if i in subgroup_indices else self.global_best_position - particle_positions[i])
                particle_velocities[i] = w * particle_velocities[i] + cognitive_velocity + social_velocity
                particle_positions[i] += particle_velocities[i]

                # Adaptive mutation based on chaos theory
                particle_positions[i] = self.chaotically_mutate(particle_positions[i], lb, ub, progress)

                # Compression: Reduce the search space dynamically with a small tweak of added resistance
                resistance = 0.01  # Fine-tuned dynamic resistance factor
                search_range_reduction = (self.budget - evaluations) / self.budget
                particle_positions[i] = np.clip(particle_positions[i],
                                                lb + search_range_reduction * (self.global_best_position - lb + resistance),
                                                ub - search_range_reduction * (ub - self.global_best_position + resistance))

        return self.global_best_position, self.global_best_value