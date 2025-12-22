import numpy as np

class AMPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.c1_start, self.c1_end = 2.5, 0.5  # adaptive cognitive component
        self.c2_start, self.c2_end = 0.5, 2.5  # adaptive social component
        self.w_start, self.w_end = 0.9, 0.4    # adaptive inertia weight
        self.global_best_position = None
        self.global_best_value = float('inf')
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        num_swarms = 3  # Divide population into multiple swarms
        swarm_size = self.population_size // num_swarms
        particle_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        particle_velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = particle_positions.copy()
        personal_best_values = np.full(self.population_size, float('inf'))

        evaluations = 0

        while evaluations < self.budget:
            for swarm_id in range(num_swarms):
                swarm_indices = range(swarm_id * swarm_size, (swarm_id + 1) * swarm_size)
                swarm_best_position, swarm_best_value = None, float('inf')

                for i in swarm_indices:
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

                    if current_value < swarm_best_value:
                        swarm_best_value = current_value
                        swarm_best_position = particle_positions[i]

                progress = evaluations / self.budget
                c1 = self.c1_start * (1 - progress) + self.c1_end * progress
                c2 = self.c2_start * (1 - progress) + self.c2_end * progress
                w = self.w_start * (1 - progress) + self.w_end * progress

                for i in swarm_indices:
                    if evaluations >= self.budget:
                        break

                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive_velocity = c1 * r1 * (personal_best_positions[i] - particle_positions[i])
                    social_velocity = c2 * r2 * (swarm_best_position - particle_positions[i])
                    particle_velocities[i] = w * particle_velocities[i] + cognitive_velocity + social_velocity
                    particle_positions[i] += particle_velocities[i]

                    # Inter-swarm Communication: occasionally share global best information
                    if np.random.rand() < 0.1:
                        particle_positions[i] += np.random.rand() * (self.global_best_position - particle_positions[i])

                    # Compression: Reduce the search space dynamically
                    search_range_reduction = (self.budget - evaluations) / self.budget
                    particle_positions[i] = np.clip(particle_positions[i],
                                                    lb + search_range_reduction * (self.global_best_position - lb),
                                                    ub - search_range_reduction * (ub - self.global_best_position))

        return self.global_best_position, self.global_best_value