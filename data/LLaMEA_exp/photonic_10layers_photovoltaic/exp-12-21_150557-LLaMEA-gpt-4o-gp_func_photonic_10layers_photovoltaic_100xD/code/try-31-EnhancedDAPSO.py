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
        self.num_subpopulations = 3  # introducing multi-population
        self.positions = [np.random.uniform(0, 1, (self.population_size, self.dim)) for _ in range(self.num_subpopulations)]
        self.velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.num_subpopulations)]
        self.personal_best_positions = [p.copy() for p in self.positions]
        self.personal_best_values = [np.full(self.population_size, float('inf')) for _ in range(self.num_subpopulations)]

    def chaotically_mutate(self, position, lb, ub, progress):
        # Add chaotic behavior to mutation
        beta = 0.5  # mutation intensity
        chaos_factor = 4 * progress * (1 - progress)  # logistic map
        mutation = beta * chaos_factor * (np.random.rand(self.dim) - 0.5)
        return np.clip(position + mutation, lb, ub)
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0

        while evaluations < self.budget:
            for idx in range(self.num_subpopulations):
                subgroup_size = max(2, int((np.sin((evaluations / self.budget) * np.pi) + 1) / 2 * self.population_size))  # dynamic subgroup size
                subgroup_indices = np.random.choice(self.population_size, subgroup_size, replace=False)
                subgroup_best_position, subgroup_best_value = None, float('inf')

                for i in range(self.population_size):
                    if evaluations >= self.budget:
                        break

                    current_value = func(self.positions[idx][i])
                    evaluations += 1

                    if current_value < self.personal_best_values[idx][i]:
                        self.personal_best_values[idx][i] = current_value
                        self.personal_best_positions[idx][i] = self.positions[idx][i]

                    if current_value < self.global_best_value:
                        self.global_best_value = current_value
                        self.global_best_position = self.positions[idx][i]

                    if i in subgroup_indices and current_value < subgroup_best_value:
                        subgroup_best_value = current_value
                        subgroup_best_position = self.positions[idx][i]

                progress = evaluations / self.budget
                c1 = self.c1_start * (1 - progress) + self.c1_end * progress
                c2 = self.c2_start * (1 - progress) + self.c2_end * progress
                w = self.w_start * (1 - progress) + self.w_end * progress

                for i in range(self.population_size):
                    if evaluations >= self.budget:
                        break

                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive_velocity = c1 * r1 * (self.personal_best_positions[idx][i] - self.positions[idx][i])
                    social_velocity = c2 * r2 * (subgroup_best_position - self.positions[idx][i] if i in subgroup_indices else self.global_best_position - self.positions[idx][i])
                    self.velocities[idx][i] = w * self.velocities[idx][i] + cognitive_velocity + social_velocity
                    self.positions[idx][i] += self.velocities[idx][i]

                    # Adaptive mutation based on chaos theory
                    self.positions[idx][i] = self.chaotically_mutate(self.positions[idx][i], lb, ub, progress)
                    
                    # Progressive search space compression
                    search_range_reduction = (self.budget - evaluations) / self.budget
                    self.positions[idx][i] = np.clip(self.positions[idx][i],
                                                     lb + search_range_reduction * (self.global_best_position - lb),
                                                     ub - search_range_reduction * (ub - self.global_best_position))
        
        return self.global_best_position, self.global_best_value