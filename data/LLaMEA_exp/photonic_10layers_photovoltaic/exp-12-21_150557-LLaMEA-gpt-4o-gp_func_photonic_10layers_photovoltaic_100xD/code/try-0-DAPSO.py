import numpy as np

class DAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w = 0.7   # inertia weight
        self.global_best_position = None
        self.global_best_value = float('inf')
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particle_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        particle_velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = particle_positions.copy()
        personal_best_values = np.full(self.population_size, float('inf'))

        evaluations = 0
        while evaluations < self.budget:
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

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - particle_positions[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - particle_positions[i])
                particle_velocities[i] = self.w * particle_velocities[i] + cognitive_velocity + social_velocity
                particle_positions[i] += particle_velocities[i]

                # Compression: Reduce the search space dynamically
                search_range_reduction = (self.budget - evaluations) / self.budget
                particle_positions[i] = np.clip(particle_positions[i],
                                                lb + search_range_reduction * (self.global_best_position - lb),
                                                ub - search_range_reduction * (ub - self.global_best_position))

        return self.global_best_position, self.global_best_value