import numpy as np

class HPESO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.c1_start, self.c1_end = 2.5, 0.5  # dynamic cognitive component
        self.c2_start, self.c2_end = 0.5, 2.5  # dynamic social component
        self.w_start, self.w_end = 0.9, 0.4    # dynamic inertia weight
        self.mutation_rate = 0.1  # rate for evolutionary mutation
        self.elitism_rate = 0.1   # elitism rate for selection
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

            progress = evaluations / self.budget
            c1 = self.c1_start * (1 - progress) + self.c1_end * progress
            c2 = self.c2_start * (1 - progress) + self.c2_end * progress
            w = self.w_start * (1 - progress) + self.w_end * progress

            num_elites = int(self.population_size * self.elitism_rate)
            elites_indices = np.argpartition(personal_best_values, num_elites)[:num_elites]
            elites_positions = personal_best_positions[elites_indices]

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                if i not in elites_indices:
                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive_velocity = c1 * r1 * (personal_best_positions[i] - particle_positions[i])
                    social_velocity = c2 * r2 * (self.global_best_position - particle_positions[i])
                    particle_velocities[i] = w * particle_velocities[i] + cognitive_velocity + social_velocity

                    # Apply mutation
                    if np.random.rand() < self.mutation_rate:
                        mutation = np.random.normal(0, 0.1 * (ub - lb), self.dim)
                        particle_positions[i] += mutation

                    particle_positions[i] += particle_velocities[i]

                    # Compression: Reduce the search space dynamically
                    search_range_reduction = (self.budget - evaluations) / self.budget
                    particle_positions[i] = np.clip(particle_positions[i],
                                                    lb + search_range_reduction * (self.global_best_position - lb),
                                                    ub - search_range_reduction * (ub - self.global_best_position))

        return self.global_best_position, self.global_best_value