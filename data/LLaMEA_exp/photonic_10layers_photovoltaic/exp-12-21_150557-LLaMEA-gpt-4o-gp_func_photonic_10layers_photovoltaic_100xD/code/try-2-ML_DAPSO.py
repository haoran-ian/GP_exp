import numpy as np

class ML_DAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.leader_count = max(2, self.population_size // 10)  # multiple leaders
        self.c1_start, self.c1_end = 2.5, 0.5  # dynamic cognitive component
        self.c2_start, self.c2_end = 0.5, 2.5  # dynamic social component
        self.w_start, self.w_end = 0.9, 0.4    # dynamic inertia weight
        self.global_best_positions = [None] * self.leader_count
        self.global_best_values = [float('inf')] * self.leader_count

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particle_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        particle_velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = particle_positions.copy()
        personal_best_values = np.full(self.population_size, float('inf'))

        evaluations = 0
        subgroup_size = self.population_size // 2  # form dynamic subgroups

        while evaluations < self.budget:
            subgroup_indices = np.random.choice(self.population_size, subgroup_size, replace=False)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(particle_positions[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particle_positions[i]

                # Update multiple leaders
                for j in range(self.leader_count):
                    if current_value < self.global_best_values[j]:
                        self.global_best_values[j] = current_value
                        self.global_best_positions[j] = particle_positions[i]
                        break

            progress = evaluations / self.budget
            c1 = self.c1_start * (1 - progress) + self.c1_end * progress
            c2 = self.c2_start * (1 - progress) + self.c2_end * progress
            w = self.w_start * (1 - progress) + self.w_end * progress

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = c1 * r1 * (personal_best_positions[i] - particle_positions[i])
                leader_index = np.random.choice(self.leader_count)
                social_velocity = c2 * r2 * (self.global_best_positions[leader_index] - particle_positions[i])
                particle_velocities[i] = w * particle_velocities[i] + cognitive_velocity + social_velocity
                particle_positions[i] += particle_velocities[i]

                # Adaptive local search for better exploitation
                if np.random.rand() < 0.1:  # 10% chance to exploit locally
                    local_search_step = 0.1 * (ub - lb) * (1 - progress)
                    particle_positions[i] += np.random.uniform(-local_search_step, local_search_step, self.dim)

                # Compression: Reduce the search space dynamically
                search_range_reduction = (self.budget - evaluations) / self.budget
                particle_positions[i] = np.clip(particle_positions[i],
                                                lb + search_range_reduction * (particle_positions[i] - lb),
                                                ub - search_range_reduction * (ub - particle_positions[i]))

        best_idx = np.argmin(self.global_best_values)
        return self.global_best_positions[best_idx], self.global_best_values[best_idx]